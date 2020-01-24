from time import sleep


class DinoAgent:
    def __init__(self, game):
        self._game = game
        self.jump()
        sleep(.5)

    def is_running(self):
        return self._game.get_playing()

    def is_crashed(self):
        return self._game.get_crashed()

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()
