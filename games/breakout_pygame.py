import sys

from ple.games.base.pygamewrapper import PyGameWrapper
from ple.games.utils.vec2d import vec2d
from ple.games.utils import percent_round_int
import pygame
from pygame.constants import K_LEFT, K_RIGHT


# gym-ple, ple を経由するので init や reset で引数指定できない
ENABLED_SHADOW_TAIL = 0


class BallShadow(pygame.sprite.Sprite):
    def __init__(self, radius, pos_init, name):
        pygame.sprite.Sprite.__init__(self)

        self.radius = int(radius * 0.75)
        self.name = name

        image = pygame.Surface((radius * 2, radius * 2))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))
        image.set_alpha(10)

        pygame.draw.circle(
            image,
            (255, 255, 255),
            (radius, radius),
            radius,
            0,
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, pos, name):
        if name == self.name:
            self.rect.center = pos


class Ball(pygame.sprite.Sprite):
    def __init__(self, radius, speed, rng,
                 pos_init, SCREEN_WIDTH, SCREEN_HEIGHT):
        pygame.sprite.Sprite.__init__(self)

        self.radius = radius
        self.speed = speed
        self.rng = rng
        self.pos = vec2d(pos_init)
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        self.pos_before = vec2d(pos_init)
        self.vel = vec2d((speed, -1.0 * speed))

        image = pygame.Surface((radius * 2, radius * 2))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.circle(
            image,
            (255, 255, 255),
            (radius, radius),
            radius,
            0,
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    # 当たり判定関数
    def line_intersection(self, p0_x, p0_y, p1_x, p1_y,
                          p2_x, p2_y, p3_x, p3_y):
        s1_x = p1_x - p0_x
        s1_y = p1_y - p0_y
        s2_x = p3_x - p2_x
        s2_y = p3_y - p2_y

        s = ((-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) /
             (-s2_x * s1_y + s1_x * s2_y))
        t = ((s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) /
             (-s2_x * s1_y + s1_x * s2_y))

        return (s >= 0 and s <= 1 and t >= 0 and t <= 1)

    def update(self, player, dt):
        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt

        radius = player.pos.y - (player.rect_height / 2)

        # Paddle との当たり判定
        if self.pos.y + self.radius >= radius:
            if self.line_intersection(self.pos_before.x,
                                      self.pos_before.y,
                                      self.pos.x,
                                      self.pos.y,
                                      (player.pos.x +
                                       player.rect_width / 2),
                                      (player.pos.y -
                                       player.rect_height / 2),
                                      (player.pos.x -
                                       player.rect_width / 2),
                                      (player.pos.y -
                                       player.rect_height / 2)):
                # 当たった場合
                self.vel.y = -1 * (self.vel.y)
                # Little randomness in order not to stuck in a static loop
                self.vel.y += self.rng.random_sample() * 0.001 - 0.0005
                self.vel.x += player.vel.x * 1.5
                self.pos.y -= self.radius

        if self.pos.y - self.radius <= 0:
            self.vel.y *= -0.99
            self.pos.y += 1.0

        if self.pos.x - self.radius <= 0:
            self.vel.x *= -0.99
            self.pos.x += 1.0

        if self.pos.x + self.radius >= self.SCREEN_WIDTH:
            self.vel.x *= -0.99
            self.pos.x -= 1.0

        self.pos_before.x = self.pos.x
        self.pos_before.y = self.pos.y

        self.rect.center = (self.pos.x, self.pos.y)


class Paddle(pygame.sprite.Sprite):
    def __init__(self, speed, rect_width, rect_height,
                 pos_init, SCREEN_WIDTH, SCREEN_HEIGHT):
        pygame.sprite.Sprite.__init__(self)

        self.speed = speed
        self.rect_height = rect_height
        self.rect_width = rect_width
        self.pos = vec2d(pos_init)
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        self.vel = vec2d((0, 0))

        image = pygame.Surface((rect_width, rect_height))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.rect(
            image,
            (255, 255, 255),
            (0, 0, rect_width, rect_height),
            0,
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, dx, dt):
        self.vel.x += dx * dt
        self.vel.x *= 0.9

        self.pos.x += dx * dt

        if self.pos.x - self.rect_height / 2 <= 0:
            self.pos.x = self.rect_height / 2
            self.vel.x = 0.0

        if self.pos.x + self.rect_height / 2 >= self.SCREEN_HEIGHT:
            self.pos.x = self.SCREEN_HEIGHT - self.rect_height / 2
            self.vel.x = 0.0

        self.rect.center = (self.pos.x, self.pos.y)


class Block(pygame.sprite.Sprite):
    def __init__(self, rect_width, rect_height,
                 pos_init, SCREEN_WIDTH, SCREEN_HEIGHT):
        pygame.sprite.Sprite.__init__(self)

        self.rect_height = rect_height
        self.rect_width = rect_width
        self.pos = vec2d(pos_init)
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        image = pygame.Surface((rect_width, rect_height))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.rect(
            image,
            (255, 255, 255),
            (0, 0, rect_width, rect_height),
            0,
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init


class Breakout_pygame(PyGameWrapper):
    def __init__(self, width=60, height=60,
                 paddle_speed_ratio=0.9, ball_speed_ratio=0.4):
        actions = {
            "left": K_LEFT,
            "right": K_RIGHT,
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)

        self.paddle_speed_ratio = paddle_speed_ratio
        self.ball_speed_ratio = ball_speed_ratio

        self.ball_radius = percent_round_int(height, 0.02)
        self.paddle_width = percent_round_int(width, 0.15)
        self.paddle_height = percent_round_int(height, 0.04)

        self.num_block_per_row = 10
        self.num_row = 6

        self.num_blocks = self.num_block_per_row * self.num_row
        self.hit_points = 5

        self.score = 0.0
        self.dy = 0.0

    def _handle_player_events(self):
        self.dx = 0

        if __name__ == "__main__":
            # for debugging mode
            pygame.event.get()
            keys = pygame.key.get_pressed()
            if keys[self.actions['left']]:
                self.dx = -self.player.speed
            elif keys[self.actions['right']]:
                self.dx = self.player.speed

            if keys[pygame.QUIT]:
                pygame.quit()
                sys.exit()
            pygame.event.pump()
        else:
            # consume events from act
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == self.actions['left']:
                        self.dx = -self.player.speed

                    if key == self.actions['right']:
                        self.dx = self.player.speed

    def getGameState(self):
        state = {
            "player_y": self.player.pos.y,
            "player_velocity": self.player.vel.y,
            "ball_x": self.ball.pos.x,
            "ball_y": self.ball.pos.y,
            "ball_velocity_x": self.ball.vel.x,
            "ball_velocity_y": self.ball.vel.y,
        }
        return state

    def getScore(self):
        return self.score

    def game_over(self):
        return (self.num_blocks < 1) or (self.hit_points < 1)

    def init(self):
        self.num_blocks = self.num_block_per_row * self.num_row
        self.hit_points = 5

        self.score = 0.0

        # player = paddle
        self.players_group = pygame.sprite.GroupSingle()
        self.player = Paddle(
            self.paddle_speed_ratio * self.height,
            self.paddle_width,
            self.paddle_height,
            (
                (self.width / 2) - (self.paddle_width / 2),
                self.height - (self.paddle_height / 2)
            ),
            self.width,
            self.height,
        )
        self.players_group.add(self.player)

        # ball
        self.ball_group = pygame.sprite.GroupSingle()
        self.ball = Ball(
            self.ball_radius,
            self.ball_speed_ratio * self.height,
            self.rng,
            (self.width - (self.width / 4), self.height / 2),
            self.width,
            self.height,
        )
        self.ball_group.add(self.ball)

        # blocks
        self.blocks_group = pygame.sprite.Group()

        block_width = self.width / self.num_block_per_row
        block_height = block_width * 0.5

        block_pos_y = self.height / 7
        block_pos_x = block_width / 2

        for row in range(self.num_row):
            for column in range(self.num_block_per_row):
                block = Block(
                            block_width,
                            block_height,
                            (block_pos_x, block_pos_y),
                            self.width,
                            self.height,
                )
                self.blocks_group.add(block)
                block_pos_x += block_width
            block_pos_x = block_width / 2
            block_pos_y += block_height

        # shadows
        self.shadows_group = pygame.sprite.Group()
        self.tail_length = 15
        self.go_back = 0

        for i in range(self.tail_length):
            shadow = BallShadow(
                self.ball_radius,
                (self.ball.pos.x, self.ball.pos.y),
                i,
            )
            self.shadows_group.add(shadow)

    def reset(self):
        self.init()
        self._reset_ball(1 if self.rng.random_sample() > 0.5 else -1)

    def _reset_ball(self, direction=-1):
        self.ball.pos.x = self.width / 2
        self.ball.pos.y = self.height / 2

        self.ball.vel.x = ((self.rng.random_sample() * self.ball.speed) -
                           self.ball.speed * 0.5)
        # self.ball.vel.y = ((self.rng.random_sample() * self.ball.speed) -
        #                    self.ball.speed * 0.5)
        self.ball.vel.y = self.ball_speed_ratio * self.height

    def step(self, dt):
        dt /= 1000.0
        self.screen.fill((0, 0, 0))

        self.player.speed = self.paddle_speed_ratio * self.height
        self.ball.speed = self.ball_speed_ratio * self.height

        # player
        self._handle_player_events()

        # ball
        self.ball.update(self.player, dt)

        # blocks
        for b in self.blocks_group.sprites():
            if self.ball.pos.y - self.ball.radius < b.pos.y + b.rect_height:
                abs_pos_x = abs(b.pos.x - self.ball.pos.x)
                if abs_pos_x < (b.rect_width / 2) + self.ball.radius:
                    self.blocks_group.remove(b)
                    self.num_blocks -= 1
                    self.score += 1.0
                    self.ball.vel.y = abs(self.ball.vel.y)

        # shadows
        self.shadows_group.update(
            (self.ball.pos.x, self.ball.pos.y),
            self.go_back
        )
        self.go_back += 1
        if self.go_back == self.tail_length:
            self.go_back = 0

        # logic
        if self.ball.pos.y >= self.height:
            self._reset_ball(1 if self.rng.random_sample() > 0.5 else -1)
            self.hit_points -= 1
        else:
            self.player.update(self.dx, dt)

        self.players_group.draw(self.screen)
        self.ball_group.draw(self.screen)
        self.blocks_group.draw(self.screen)
        if ENABLED_SHADOW_TAIL:
            self.shadows_group.draw(self.screen)

    def _print_info(self):
        print("score:", self.getScore(), "d:", self.game_over())


if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = Breakout_pygame(width=200, height=200)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)

    for eps in range(1, 6):
        score = 0.0
        game.reset()
        while not game.game_over():
            dt = game.clock.tick_busy_loop(60)
            game.step(dt)
            score += game.getScore()
            print("eps:{}, score: {}, HP: {}".format(
                eps, score, game.hit_points))
            pygame.display.update()
