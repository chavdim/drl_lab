import sys

import pygame
from pygame.constants import K_w, K_s
from ple.games.base.pygamewrapper import PyGameWrapper
from ple.games.utils.vec2d import vec2d
from ple.games.utils import percent_round_int


class BallShadow(pygame.sprite.Sprite):
    def __init__(self, radius, speed, rng,
                 pos_init, SCREEN_WIDTH, SCREEN_HEIGHT, init_x, init_y):

        pygame.sprite.Sprite.__init__(self)

        self.radius = int(radius*0.75)
        self.speed = speed
        self.pos = vec2d((init_x, init_y))
        self.pos_before = vec2d((init_x, init_y))

        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        image = pygame.Surface((radius * 2, radius * 2))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.circle(
            image,
            (122, 255, 255),
            (radius, radius),
            radius,
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init


class Ball(pygame.sprite.Sprite):

    def __init__(self, radius, speed, rng,
                 pos_init, SCREEN_WIDTH, SCREEN_HEIGHT):

        pygame.sprite.Sprite.__init__(self)

        self.rng = rng
        self.radius = radius
        self.speed = speed
        self.pos = vec2d(pos_init)
        self.pos_before = vec2d(pos_init)
        self.vel = vec2d((speed, -1.0 * speed))

        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        image = pygame.Surface((radius * 2, radius * 2))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.circle(
            image,
            (255, 255, 255),
            (radius, radius),
            radius,
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

        self.playerHit = False

    def line_intersection(self, p0_x, p0_y, p1_x,
                          p1_y, p2_x, p2_y, p3_x, p3_y):

        s1_x = p1_x - p0_x
        s1_y = p1_y - p0_y
        s2_x = p3_x - p2_x
        s2_y = p3_y - p2_y

        s = ((-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) /
             (-s2_x * s1_y + s1_x * s2_y))
        t = ((s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) /
             (-s2_x * s1_y + s1_x * s2_y))

        return (s >= 0 and s <= 1 and t >= 0 and t <= 1)

    def update(self, agentPlayer, dt):

        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt

        is_pad_hit = False
        ####
        self.playerHit = False

        radius = agentPlayer.pos.y-(agentPlayer.rect_height/2)
        if self.pos.y+self.radius >= radius:
            if self.line_intersection(self.pos_before.x, self.pos_before.y,
                                      self.pos.x, self.pos.y,
                                      (agentPlayer.pos.x +
                                       agentPlayer.rect_width/2),
                                      (agentPlayer.pos.y -
                                       agentPlayer.rect_height/2),
                                      (agentPlayer.pos.x -
                                       agentPlayer.rect_width/2),
                                      (agentPlayer.pos.y -
                                       agentPlayer.rect_height/2)):
                # self.pos.x = max(0, self.pos.x)
                # exchaged x,y
                self.vel.y = -1 * (self.vel.y)  # + self.speed * 0.01)
                self.vel.x += agentPlayer.vel.x * 1.5
                self.pos.y -= self.radius
                is_pad_hit = True
                self.playerHit = True

        """
        if self.pos.x >= cpuPlayer.pos.x - cpuPlayer.rect_width:
            if self.line_intersection(self.pos_before.x, self.pos_before.y,
                                      self.pos.x, self.pos.y,
                                      (cpuPlayer.pos.x -
                                       cpuPlayer.rect_width/2),
                                      (cpuPlayer.pos.y -
                                       cpuPlayer.rect_height/2),
                                      (cpuPlayer.pos.x -
                                       cpuPlayer.rect_width/2),
                                      (cpuPlayer.pos.y +
                                       cpuPlayer.rect_height/2)):
                self.pos.x = min(self.SCREEN_WIDTH, self.pos.x)
                self.vel.x = -1 * (self.vel.x + self.speed * 0.05)
                self.vel.y += cpuPlayer.vel.y * 0.006
                self.pos.x -= self.radius
                is_pad_hit = True
        """

        # Little randomness in order not to stuck in a static loop
        if is_pad_hit:
            self.vel.y += self.rng.random_sample() * 0.001 - 0.0005

        if self.pos.y - self.radius <= 0:
            self.vel.y *= -0.99
            self.pos.y += 1.0

        """
        if self.pos.y + self.radius >= self.SCREEN_WIDTH:
            self.vel.y *= -0.99
            self.pos.y -= 1.0
        """

        if self.pos.x - self.radius <= 0:
            self.vel.x *= -0.99
            self.pos.x += 1.0

        if self.pos.x + self.radius >= self.SCREEN_WIDTH:
            self.vel.x *= -0.99
            self.pos.x -= 1.0

        # pygame.draw.line(self.image,(122,255,255),( 0, 0),
        #    ( 30,30),2)

        self.pos_before.x = self.pos.x
        self.pos_before.y = self.pos.y

        self.rect.center = (self.pos.x, self.pos.y)


class Player(pygame.sprite.Sprite):
    def __init__(self, speed, rect_width, rect_height,
                 pos_init, SCREEN_WIDTH, SCREEN_HEIGHT, isPlayer=True):

        pygame.sprite.Sprite.__init__(self)

        self.speed = speed
        self.pos = vec2d(pos_init)
        self.vel = vec2d((0, 0))

        self.rect_height = rect_height
        self.rect_width = rect_width
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        image = pygame.Surface((rect_width, rect_height))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        if isPlayer:
            pygame.draw.rect(
                image,
                (255, 255, 255),
                (0, 0, rect_width, rect_height),
                0
            )
        else:
            pygame.draw.rect(
                image,
                (255, 255, 255),
                (0, 0, rect_width, rect_height),
                0
            )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, dx, dt):
        self.vel.x += dx * dt
        self.vel.x *= 0.9

        # self.pos.y += self.vel.y
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

        self.pos = vec2d(pos_init)

        # rect_height-=2 #buffers
        # rect_width-=2

        self.rect_height = rect_height
        self.rect_width = rect_width
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.dead = 0

        image = pygame.Surface((rect_width, rect_height))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.rect(
            image,
            (255, 255, 255),
            (0, 0, rect_width, rect_height),
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, dx, dt):
        pass


class Breakout_pygame(PyGameWrapper):
    """
    Loosely based on code from marti1125's `pong game`_.

    .. _pong game: https://github.com/marti1125/pong/

    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    MAX_SCORE : int (default: 11)
        The max number of points the agent or cpu need to score to cause
        a terminal state.

    cpu_speed_ratio: float (default: 0.5)
        Speed of opponent (useful for curriculum learning)

    players_speed_ratio: float (default: 0.25)
        Speed of player (useful for curriculum learning)

    ball_speed_ratio: float (default: 0.75)
        Speed of ball (useful for curriculum learning)

    """

    def __init__(self, width=60, height=60, cpu_speed_ratio=0.7,
                 players_speed_ratio=0.9, ball_speed_ratio=0.4,  MAX_SCORE=1):

        actions = {
            "up": K_w,
            "down": K_s
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)

        # the %'s come from original values, wanted to keep same ratio when you
        # increase the resolution.
        self.screen_height = height
        self.ball_radius = percent_round_int(height, 0.02)

        self.cpu_speed_ratio = cpu_speed_ratio
        self.ball_speed_ratio = ball_speed_ratio
        self.players_speed_ratio = players_speed_ratio

        # self.paddle_width = percent_round_int(width, 0.023)
        self.paddle_width = percent_round_int(width, 0.15)
        self.paddle_height = percent_round_int(height, 0.04)
        # self.paddle_dist_to_wall = percent_round_int(width, 0.0625)
        self.paddle_dist_to_wall = 0

        self.numBlockPerRow = 10
        self.numRow = 6
        self.MAX_SCORE = self.numBlockPerRow * self.numRow

        self.dy = 0.0
        # need to deal with MAX_SCORE on either side winning
        self.score_sum = 0.0
        self.score_counts = {
            "agent": 0.0,
            "cpu": 0.0
        }

    def _handle_player_events(self):
        self.dx = 0

        if __name__ == "__main__":
            # for debugging mode
            pygame.event.get()
            keys = pygame.key.get_pressed()
            if keys[self.actions['up']]:
                self.dx = -self.agentPlayer.speed
            elif keys[self.actions['down']]:
                self.dx = self.agentPlayer.speed

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
                    if key == self.actions['up']:
                        self.dx = -self.agentPlayer.speed

                    if key == self.actions['down']:
                        self.dx = self.agentPlayer.speed

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        Returns
        -------

        dict
            * player y position.
            * players velocity.
            * cpu y position.
            * ball x position.
            * ball y position.
            * ball x velocity.
            * ball y velocity.

            See code for structure.

        """
        state = {
            "player_y": self.agentPlayer.pos.y,
            "player_velocity": self.agentPlayer.vel.y,
            # "cpu_y": self.cpuPlayer.pos.y,
            "ball_x": self.ball.pos.x,
            "ball_y": self.ball.pos.y,
            "ball_velocity_x": self.ball.vel.x,
            "ball_velocity_y": self.ball.vel.y
        }

        return state

    def getScore(self):
        return self.score_sum
        # return self.tickScore

    def game_over(self):
        # pong used 11 as max score
        return (self.score_counts['agent'] == self.MAX_SCORE)
        # or (self.score_counts['cpu'] == self.MAX_SCORE)

    def init(self):
        self.score_counts = {
            "agent": 0.0,
            "cpu": 0.0
        }

        self.score_sum = 0.0
        self.tickScore = 0.0
        # shadows
        self.goback = 0
        self.tailLen = 15
        self.tailSpeed = 1.5
        self.shadows = []
        self.steps = 0

        self.ball = Ball(
            self.ball_radius,
            self.ball_speed_ratio * self.height,
            self.rng,
            (self.width - (self.width/4), self.height / 2),
            self.width,
            self.height
        )

        self.agentPlayer = Player(
            self.players_speed_ratio * self.height,
            self.paddle_width,
            self.paddle_height,
            (((self.width/2) - (self.paddle_width/2)),
             self.height - (self.paddle_height/2)),
            self.width,
            self.height)

        # blocks
        self.blocks_group = pygame.sprite.Group()
        # self.blocks=[]
        # buff = self.width/100
        numBlockPerRow = 10
        numRow = 6

        blockW = self.width/numBlockPerRow
        blockH = blockW*0.5

        rd = self.height/7  # space above
        cd = blockW/2
        for row in range(numRow):

            for column in range(numBlockPerRow):
                block = Block(blockW, blockH, (cd, rd),
                              self.width, self.height)
                self.blocks_group.add(block)
                cd += blockW  # +buff
            cd = blockW/2
            rd += blockH  # +buff

        self.players_group = pygame.sprite.Group()
        self.players_group.add(self.agentPlayer)

        self.ball_group = pygame.sprite.Group()

        self.ball_group.add(self.ball)

    def reset(self):
        self.init()
        # after game over set random direction of ball otherwise it will
        # always be the same
        self._reset_ball(1 if self.rng.random_sample() > 0.5 else -1)
        # self._reset_ball(-1)

    def _reset_ball(self, direction=-1):
        self.ball.pos.x = self.width/2  # move it to the center
        self.ball.pos.y = self.height/2

        # we go in the same direction that they lost in but at starting vel.
        self.ball.vel.x = ((self.rng.random_sample() * self.ball.speed) -
                           self.ball.speed * 0.5)
        # self.ball.vel.y = ((self.rng.random_sample() * self.ball.speed) -
        #                    self.ball.speed * 0.5)
        self.ball.vel.y = self.ball_speed_ratio*self.height

    def step(self, dt):
        self.steps += 1
        dt /= 1000.0
        self.screen.fill((0, 0, 0))

        self.agentPlayer.speed = self.players_speed_ratio * self.height

        self.ball.speed = self.ball_speed_ratio * self.height

        self._handle_player_events()

        # doesnt make sense to have this, but include if needed.
        self.score_sum += self.rewards["tick"]
        self.score_sum -= self.rewards["tick"]

        self.tickScore = 0.0

        self.ball.update(self.agentPlayer, dt)
        # shadows

        if len(self.shadows) < self.tailLen:
            shadow = BallShadow(
                        self.ball_radius,
                        self.ball_speed_ratio * self.height,
                        self.rng,
                        (self.width / 2, self.height / 2),
                        self.width,
                        self.height,
                        self.ball.pos.x,
                        self.ball.pos.y
                    )
            # self.players_group.add(shadow)
            self.shadows.append(shadow)
        else:
            self.shadows[self.goback].pos.x = self.ball.pos.x
            self.shadows[self.goback].pos.y = self.ball.pos.y
            self.shadows[self.goback].rect.center = (
                self.shadows[self.goback].pos.x,
                self.shadows[self.goback].pos.y)
            self.goback += 1
            if self.goback == self.tailLen:
                self.goback = 0

        # reward for hitting
        # if self.ball.playerHit == True:
        #    self.score_sum += 0.5

        is_terminal_state = False

        # blocks
        for b in self.blocks_group.sprites():
            if self.ball.pos.y-self.ball.radius < b.pos.y+b.rect_height:
                abs_pos_x = abs(b.pos.x-self.ball.pos.x)
                if abs_pos_x < b.rect_width/2 + self.ball.radius:
                    self.blocks_group.remove(b)
                    # reward
                    self.score_sum += self.rewards["positive"]
                    # self.tickScore =self.rewards["positive"]
                    self.score_counts["agent"] += 1.0

                    self.ball.vel.y = abs(self.ball.vel.y)

        # logic
        if self.ball.pos.y >= self.screen_height:
            self.score_sum += self.rewards["negative"]
            self.tickScore = -1

            self.score_counts["cpu"] += 1.0
            self._reset_ball(1)
            is_terminal_state = True

        if is_terminal_state:
            # winning
            if self.score_counts['agent'] == self.MAX_SCORE:
                self.score_sum += self.rewards["win"]
                self.tickScore = self.rewards["positive"]

            # losing
            if self.score_counts['cpu'] == self.MAX_SCORE:
                self.score_sum += self.rewards["loss"]
        else:
            self.agentPlayer.update(self.dx, dt)
            # self.cpuPlayer.updateCpu(self.ball, dt)

        self.players_group.draw(self.screen)
        self.ball_group.draw(self.screen)
        self.blocks_group.draw(self.screen)


if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = Breakout_pygame(width=100, height=100)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(60)
        game.step(dt)
        pygame.display.update()
