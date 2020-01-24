from game import Game
from agent import DinoAgent
from game_state import GameState
from model import build_model, train_network


def play_game(observe=False):
    game = Game()
    dino = DinoAgent(game)
    game_state = GameState(dino, game)
    model = build_model()
    train_network(model, game_state)


if __name__ == '__main__':
    play_game()
