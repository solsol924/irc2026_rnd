import cv2
import numpy as np
import random
import time

class Poker:
    def __init__(self):
        self.load_cards()

    def load_cards(self):
        self.deck = []

        img = cv2.imread("/home/noh/Downloads/cards2048.png")
        h, w, _ = img.shape
        print("=" * 40)
        print(f"Image size: {w} x {h}")

        columns = 13
        rows = 5
        card_w = w // columns
        card_h = h // rows

        suits = ["♠", "♦", "♥", "♣"]
        numbers = ["A"] + [str(n) for n in range(2, 11)] + ["J", "Q", "K"]

        for row, suit in enumerate(suits):
            for col, num in enumerate(numbers):
                x1, y1 = col * card_w + col // 2, row * card_h + row // 2
                x2, y2 = x1 + card_w, y1 + card_h
                card_img = img[y1:y2, x1:x2]
                name = f"{suit}{num}"
                self.deck.append((name, card_img))

        # 조커 + 뒷면
        joker_y1 = h - card_h
        joker_y2 = h
        joker1 = img[joker_y1:joker_y2, 0:card_w]
        joker2 = img[joker_y1:joker_y2, card_w:2 * card_w]
        back_y1 = h - card_h
        back_y2 = h
        back = img[back_y1:back_y2, 2 * card_w + 1:3 * card_w + 1]
        self.deck.append(("🃏(B)", joker1))
        self.deck.append(("🃏(R)", joker2))
        self.back_img = back

        print(f"Total cards: {len(self.deck)}")
        print("=" * 40)

    def shuffle(self):
        random.shuffle(self.deck)

    def deal(self, n_players=4, n_cards=7):
        hands = [[] for _ in range(n_players)]

        card_h, card_w, _ = self.back_img.shape
        card_w //= 2
        card_h //= 2
        margin_x = 20
        margin_y = 30
        board_w = n_cards * (card_w + margin_x) + 250
        board_h = n_players * (card_h + margin_y) + 100
        board = np.zeros((board_h, board_w, 3), dtype=np.uint8)

        for i in range(n_players):
            y = 50 + i * (card_h + margin_y)
            cv2.putText(board, f"Player{i + 1}", (60, y + card_h // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Poker Table", board)
        cv2.waitKey(500)

        # 한 장씩 돌아가며 배포
        for j in range(n_cards):
            for i in range(n_players):
                if not self.deck:
                    break
                name, img = self.deck.pop(0)
                hands[i].append((name, img))

                y = 50 + i * (card_h + margin_y)
                x = 220 + j * (card_w + margin_x)

                back_resized = cv2.resize(self.back_img, (card_w, card_h))
                board[y:y + card_h, x:x + card_w] = back_resized
                cv2.imshow("Poker Table", board)
                cv2.waitKey(150)

        # 뒤집기
        time.sleep(0.5)
        for i in range(n_players):
            for j in range(n_cards):
                name, img = hands[i][j]
                resized = cv2.resize(img, (card_w, card_h))
                y = 50 + i * (card_h + margin_y)
                x = 220 + j * (card_w + margin_x)
                board[y:y + card_h, x:x + card_w] = resized
                cv2.imshow("Poker Table", board)
                cv2.waitKey(100)

        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        # 콘솔 출력
        print("=" * 40)
        print(f"Remaining cards: {len(self.deck)}")
        print("=" * 40)
        for i, cards in enumerate(hands):
            print(f"Player {i+1}")
            self.show_cards(cards)
            print(f"{len(self.deck)} cards left in deck")
            print("-" * 40 if i < n_players - 1 else "=" * 40)

    def show_cards(self, cards):
        for name, _ in cards:
            print(f"[{name}] ", end="")
        print()


# 🧩 regame 입력으로 반복 실행
if __name__ == "__main__":
    while True:
        poker = Poker()
        poker.shuffle()
        poker.deal(4, 7)

        cmd = input("👉 다시 하려면 'regame' 입력 (그 외 입력 시 종료): ").strip().lower()
        if cmd != "regame":
            print("게임을 종료합니다 🎮")
            break
        print("\n새 게임을 시작합니다...\n")
