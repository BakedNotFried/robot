import pygame

NORMAL = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (255, 0, 255)

KEY_START_CONTROL = pygame.K_q
KEY_START_RECORDING = pygame.K_w
KEY_QUIT_RECORDING = pygame.K_e
KEY_SAVE_RECORDING = pygame.K_r
KEY_DISCARD_RECORDING = pygame.K_t

KEY_LOCK_ROBOT = pygame.K_a
KEY_UNLOCK_ROBOT = pygame.K_s

START = 0
RECORDING = 1
QUIT = 2
SAVE = 3
DISCARD = 4


class KeyboardInterface:
    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((800, 800))
        self._set_color(NORMAL)

        self.record_state = None
        self.lock_robot = True


    def update(self) -> str:
        pressed_last = self._get_pressed()

        if KEY_LOCK_ROBOT in pressed_last:
            self.lock_robot = True
        elif KEY_UNLOCK_ROBOT in pressed_last:
            self.lock_robot = False
        elif KEY_START_CONTROL in pressed_last:
            self._set_color(GREEN)
            self.record_state = START
        elif KEY_START_RECORDING in pressed_last:
            self._set_color(BLUE)
            self.record_state = RECORDING
        elif KEY_QUIT_RECORDING in pressed_last:
            self._set_color(RED)
            self.record_state = QUIT
        elif KEY_SAVE_RECORDING in pressed_last:
            self._set_color(PURPLE)
            self.record_state = SAVE
        elif KEY_DISCARD_RECORDING in pressed_last:
            self._set_color(NORMAL)
            self.record_state = DISCARD


    def _get_pressed(self):
        pressed = []
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pressed.append(event.key)
        return pressed


    def _set_color(self, color):
        self._screen.fill(color)
        pygame.display.flip()


def main():
    kb = KBInterface()
    while True:
        kb.update()
        if kb.record_state == START:
            print("Start")
        elif kb.record_state == RECORDING:
            print("Recording")
        elif kb.record_state == QUIT:
            print("Quit")
        elif kb.record_state == SAVE:
            print("Save")
        elif kb.record_state == DISCARD:
            print("Discard")
        if kb.lock_robot:
            print("Robot Locked")
        elif not kb.lock_robot:
            print("Robot Unlocked")


if __name__ == "__main__":
    main()
