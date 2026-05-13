#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <robot_msgs/msg/line_point.hpp>
#include <robot_msgs/msg/line_points_array.hpp>
#include <robot_msgs/msg/motion_end.hpp>   // #################################################################### 판단 프레임 수 바꿀 때 line_sub 도 고려해라~~~
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "yolo.hpp"  // YoloTRT

#include <vector>
#include <tuple>
#include <unordered_map>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>

using std::placeholders::_1;
using LinePoint = robot_msgs::msg::LinePoint;
using LinePointsArray = robot_msgs::msg::LinePointsArray;
using MotionEnd = robot_msgs::msg::MotionEnd;
using my_cv::Detection;
using my_cv::YoloTRT;

class RectangleTracker {
public:
    RectangleTracker(int max_lost, float max_dist, int min_found)
        : max_lost_(max_lost), max_dist_(max_dist), min_found_(min_found), next_id_(0) {}

    // ★ 트래커 리셋: 수집 시작 시 잔상 제거용
    void reset() {
        rectangles_.clear();
        next_id_ = 0;
    }

    std::unordered_map<int, std::tuple<int, int, int, int>> update(const std::vector<std::pair<int, int>>& centers) {
        std::set<int> matched_ids;
        std::unordered_map<int, std::tuple<int, int, int, int>> updated;

        for (const auto& [cx, cy] : centers) {
            int best_id = -1;
            float best_dist = max_dist_;

            for (const auto& [id, data] : rectangles_) {
                if (matched_ids.count(id)) continue;
                auto [px, py, lost, found] = data;
                float dist = std::hypot(cx - px, cy - py);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_id = id;
                }
            }

            if (best_id != -1) {
                auto [_, __, ___, found] = rectangles_[best_id];
                updated[best_id] = {cx, cy, 0, std::min(found + 1, min_found_)};
                matched_ids.insert(best_id);
            } else {
                updated[next_id_++] = {cx, cy, 0, 0};
            }
        }

        for (const auto& [id, data] : rectangles_) {
            if (!matched_ids.count(id)) {
                auto [cx, cy, lost, found] = data;
                if (++lost < max_lost_) {
                    updated[id] = {cx, cy, lost, found};
                }
            }
        }

        rectangles_ = updated;

        std::unordered_map<int, std::tuple<int, int, int, int>> valid;
        for (const auto& [id, data] : rectangles_) {
            auto [cx, cy, lost, found] = data;
            if (found >= min_found_) {
                valid[id] = data;
            }
        }
        return valid;
    }

private:
    int max_lost_, min_found_, next_id_;
    float max_dist_;
    std::unordered_map<int, std::tuple<int, int, int, int>> rectangles_;
};

class YoloCppNode : public rclcpp::Node {
public:
    YoloCppNode()
        : Node("yolo_cpp_node"),
          model_("/home/rnd/rnd/best.engine"),
          tracker_(1, 60.0, 1), // lost, max_len, found
          time_text_(""),
          collecting_(false),
          armed_(false),
          frames_left_(0),
          max_len_(0),
          frame_idx_(0),
          window_id_(0)
    {
        // 파라미터: 수집 프레임 길이(파이썬 max_len과 맞춤)
        this->declare_parameter<int>("max_len", 15);
        max_len_ = this->get_parameter("max_len").as_int();

        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/cam2/color/image_raw", 10,
            std::bind(&YoloCppNode::image_callback, this, _1));

        publisher_ = this->create_publisher<LinePointsArray>("candidates", 10);

        // ★ 모션 종료 신호 구독
        motion_sub_ = this->create_subscription<MotionEnd>(
            "/motion_end", 10,
            std::bind(&YoloCppNode::motion_callback, this, _1));

        last_report_time_ = this->get_clock()->now();
        frame_count_ = 0;
        total_time_ = 0.0;

        cv::namedWindow("YOLO Detection", cv::WINDOW_NORMAL);
    }

private:
    // ★ MotionEnd 콜백: true(=모션 종료)면 수집 윈도우 시작
    void motion_callback(const MotionEnd::SharedPtr msg) {
        bool motion_end = static_cast<bool>(msg->motion_end_detect);
        if (motion_end && !collecting_ && !armed_) {
            armed_ = true;
            RCLCPP_INFO(this->get_logger(), "[YOLO] I got motion_end !!");
        }
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        auto start_time = this->get_clock()->now();

        auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        cv::Mat bgr = cv_ptr->image;
        
        int h = bgr.rows, w = bgr.cols;
        int x1 = 0 * w / 5, x2 = w * 5 / 5, y1 = 0 * h / 12, y2 = 12 * h / 12; // ROI (파이썬과 동일)
        
        if (armed_ && !collecting_){
            collecting_ = true;
            armed_ = false;
            frames_left_ = max_len_;
            frame_idx_ = 0;
            window_id_ += 1;
            tracker_.reset();  // 수집 시작 시 트래커 초기화
            RCLCPP_INFO(this->get_logger(), "[YOLO] Window %d start: total = %d (wait 5, publish 9)", window_id_, max_len_);
        }

        if (collecting_){

            frame_idx_ += 1;
            bool do_infer = (frame_idx_ >= 6);
            bool do_publish = (frame_idx_ >= 7);

            std::vector<std::tuple<int,int,int,int>> items;

            if (do_infer){
                auto t1 = this->get_clock()->now(); // 1. image

                std::vector<Detection> detections;
                model_.Infer(bgr, detections, bgr.cols, bgr.rows);
                auto t2 = this->get_clock()->now(); // 2. YOLOv5

                std::vector<std::pair<int, int>> centers;
                centers.reserve(detections.size());
                for (const auto& det : detections) {
                    if (det.class_id != 0) continue;

                    cv::Rect box = det.box;
                    int cx = box.x + box.width / 2;
                    int cy = box.y + box.height / 2;

                    if (cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2) {
                        centers.emplace_back(cx, cy);
                        cv::rectangle(bgr, box, cv::Scalar(0, 255, 0), 2);
                        cv::circle(bgr, {cx, cy}, 3, cv::Scalar(0, 0, 255), -1);
                    }
                }

                auto valid = tracker_.update(centers);
                items.reserve(valid.size());
                for (const auto& [id, data] : valid) {
                    auto [cx, cy, lost, found] = data;
                    items.emplace_back(cx, cy, lost, found);
                }

                std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
                    if (std::get<1>(a) != std::get<1>(b)) return std::get<1>(a) > std::get<1>(b); // y desc
                    if (std::get<2>(a) != std::get<2>(b)) return std::get<2>(a) < std::get<2>(b); // lost asc
                    return std::get<3>(a) > std::get<3>(b);                                       // found desc
                });
                constexpr size_t kKeep = 5;
                if (items.size() > kKeep) items.resize(kKeep);

                auto t3 = this->get_clock()->now(); // 3. sort
                
                if (do_publish){
                    LinePointsArray msg_array;
                    msg_array.points.reserve(items.size());
                    for (const auto& t : items) {
                        LinePoint p;
                        p.cx   = std::get<0>(t);
                        p.cy   = std::get<1>(t);
                        p.lost = std::get<2>(t);
                        msg_array.points.push_back(p);
                    }
                    publisher_->publish(msg_array);
                }

                auto t4 = this->get_clock()->now(); // 4. publish
                // 시간 디버깅
                std::cout << std::fixed;
                std::cout << "[CPP(ms)] Total: " << (t4 - start_time).seconds() * 1000.0 
                << " | Image: " << (t1 - start_time).seconds() * 1000.0 
                << " | Inference: " << (t2 - t1).seconds() * 1000.0 
                << " | Tracking: " << (t3 - t2).seconds() * 1000.0 
                << " | Publish: " << (t4 - t3).seconds() * 1000.0 << std::endl;
            }
            else {
                cv::putText(bgr, "Waiting,,,", {10, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,255}, 2);
            }

            frames_left_ -= 1;

            // 15프레임 끝났으면
            if (frames_left_ <= 0) {
                collecting_ = false;
                frames_left_ = 0;
                RCLCPP_INFO(this->get_logger(), "[YOLO] Window %d done at frame = %d", window_id_, frame_idx_);
            }      
        }

        else {
            cv::putText(bgr, "In move", {10, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,255}, 2);
        }
        
        cv::rectangle(bgr, cv::Rect(x1, y1, x2 - x1, y2 - y1), cv::Scalar(0, 255, 0), 1);
        // 딜레이, FPS 계산
        rclcpp::Time now = this->get_clock()->now();
        rclcpp::Duration elapsed = now - last_report_time_;
        frame_count_++;
        total_time_ += (now - start_time).seconds();

        if (elapsed.seconds() >= 1.0) {
            double avg_ping = total_time_ / frame_count_;
            double fps = frame_count_ / elapsed.seconds();
            last_report_time_ = now;
            frame_count_ = 0;
            total_time_ = 0.0;
            time_text_ = "PING: " + std::to_string(avg_ping * 1000.0).substr(0, 6) + "ms | FPS: " + std::to_string(fps).substr(0, 5);
        }
        cv::putText(bgr, time_text_, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,0}, 2);
        cv::imshow("YOLO Detection", bgr);
        cv::waitKey(1);

    }
    
    // subs / pubs
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Subscription<MotionEnd>::SharedPtr motion_sub_;            // ★ 추가
    rclcpp::Publisher<LinePointsArray>::SharedPtr publisher_;
    // model / tracker
    YoloTRT model_;
    RectangleTracker tracker_;
    // timing
    rclcpp::Time last_report_time_;
    int frame_count_;
    double total_time_;
    std::string time_text_;
    // collecting window
    bool collecting_;
    bool armed_; // 방어!
    int frames_left_;
    int max_len_;
    int frame_idx_;
    int window_id_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YoloCppNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}