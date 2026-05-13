from setuptools import find_packages, setup

package_name = 'rnd'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jinsoo',
    maintainer_email='jinsoo@todo.todo',
    description='RealSense Image Subscriber Node',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # '실행할_이름 = 패키지명.파일명:함수명'
            'realsense_test = rnd.realsense_test:main',
            'depth = rnd.depth:main',
            'color_mask_test = rnd.color_mask_test:main',
            'ball_detect = rnd.ball_detect:main',
            'detect_line = rnd.detect_line:main',
            'sol_line_publisher = rnd.sol_line_publisher:main',
            'detect_ball = rnd.detect_ball:main',
            'expo = rnd.expo:main',
            'detect_line_2 = rnd.detect_line_2:main',
            'track_line_main = rnd.track_line_main:main',
            'yolo_detector = rnd.yolo_detector:main',
            'decision_line_track = rnd.decision_line_track:main',
        ],
    },
)
