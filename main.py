# import pygame
import cv2
import paddlex
from paddlex.cls import transforms
# from sprites import *
from utils import *
from tank_war import *
import imutils
import numpy as np


class TankWar:
    train_transforms = transforms.Compose([
        transforms.RandomCrop(crop_size=224),
        transforms.Normalize()
    ])
    global bg
    bg = None
    model = paddlex.load_model('weights/final')

    def process_gesture(self, thresholded):

        bg = None
        input_im = cv2.merge(
            [thresholded, thresholded, thresholded])
        result = self.model.predict(
            input_im, topk=5, transforms=self.train_transforms)
        gesture = result[0]['category']
        cv2.putText(input_im, gesture, (0, 20),
                    cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)

        layout = np.zeros(input_im.shape)
        final = []
        for clas in CLASSES:
            for v in result:
                if v['category'] == clas:
                    final.append(v['score'])
                    break

        for (i, score) in enumerate(final):
            # construct the label text
            text = "{}: {:.2f}%".format(CLASSES[i], score * 100)

            w = int(score * 300)
            cv2.rectangle(layout, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(layout, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)

        return gesture, input_im, layout

    def run_avg(self, image, aWeight):
        global bg
        if bg is None:
            bg = image.copy().astype('float')
            return

        cv2.accumulateWeighted(image, bg, aWeight)

    def segment(self, image, threshold=25):
        global bg
        diff = cv2.absdiff(bg.astype('uint8'), image)

        thresholded = cv2.threshold(diff,
                                    threshold,
                                    255,
                                    cv2.THRESH_BINARY)[1]

        (cnts, _) = cv2.findContours(thresholded.copy(),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) == 0:
            return
        else:
            segmented = max(cnts, key=cv2.contourArea)
            return (thresholded, segmented)

    def __init__(self):
        self.screen = pygame.display.set_mode(Settings.SCREEN_RECT.size)
        self.clock = pygame.time.Clock()
        self.game_still = True
        self.hero = None
        self.enemies = None
        self.enemy_bullets = None
        self.walls = None

    @staticmethod
    def __init_game():
        """
        初始化游戏的一些设置
        :return:
        """
        pygame.init()  # 初始化pygame模块
        pygame.display.set_caption(Settings.GAME_NAME)  # 设置窗口标题
        pygame.mixer.init()  # 初始化音频模块




    def __create_sprite(self):
        self.hero = Hero(Settings.HERO_IMAGE_NAME, self.screen)
        self.enemies = pygame.sprite.Group()
        self.enemy_bullets = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        for i in range(Settings.ENEMY_COUNT):
            direction = random.randint(0, 3)
            enemy = Enemy(Settings.ENEMY_IMAGES[direction], self.screen)
            enemy.direction = direction
            self.enemies.add(enemy)
        self.__draw_map()

    def __draw_map(self):
        """
        绘制地图
        :return:
        """
        for y in range(len(Settings.MAP_ONE)):
            for x in range(len(Settings.MAP_ONE[y])):
                if Settings.MAP_ONE[y][x] == 0:
                    continue
                wall = Wall(Settings.WALLS[Settings.MAP_ONE[y][x]], self.screen)
                wall.rect.x = x * Settings.BOX_SIZE
                wall.rect.y = y * Settings.BOX_SIZE
                if Settings.MAP_ONE[y][x] == Settings.RED_WALL:
                    wall.type = Settings.RED_WALL
                elif Settings.MAP_ONE[y][x] == Settings.IRON_WALL:
                    wall.type = Settings.IRON_WALL
                elif Settings.MAP_ONE[y][x] == Settings.WEED_WALL:
                    wall.type = Settings.WEED_WALL
                elif Settings.MAP_ONE[y][x] == Settings.SEA_WALL:
                    wall.type = Settings.SEA_WALL
                elif Settings.MAP_ONE[y][x] == Settings.BOSS_WALL:
                    wall.type = Settings.BOSS_WALL
                    wall.life = 1
                self.walls.add(wall)

    def __check_keydown(self, event):
        """检查按下按钮的事件"""
        if event.key == pygame.K_LEFT:
            # 按下左键
            self.hero.direction = Settings.LEFT
            self.hero.is_moving = True
            self.hero.is_hit_wall = False
        elif event.key == pygame.K_RIGHT:
            # 按下右键
            self.hero.direction = Settings.RIGHT
            self.hero.is_moving = True
            self.hero.is_hit_wall = False
        elif event.key == pygame.K_UP:
            # 按下上键
            self.hero.direction = Settings.UP
            self.hero.is_moving = True
            self.hero.is_hit_wall = False
        elif event.key == pygame.K_DOWN:
            # 按下下键
            self.hero.direction = Settings.DOWN
            self.hero.is_moving = True
            self.hero.is_hit_wall = False
        elif event.key == pygame.K_SPACE:
            # 坦克发子弹
            self.hero.shot()

    def __check_keyup(self, event):
        """检查松开按钮的事件"""
        if event.key == pygame.K_LEFT:
            # 松开左键
            self.hero.direction = Settings.LEFT
            self.hero.is_moving = False
        elif event.key == pygame.K_RIGHT:
            # 松开右键
            self.hero.direction = Settings.RIGHT
            self.hero.is_moving = False
        elif event.key == pygame.K_UP:
            # 松开上键
            self.hero.direction = Settings.UP
            self.hero.is_moving = False
        elif event.key == pygame.K_DOWN:
            # 松开下键
            self.hero.direction = Settings.DOWN
            self.hero.is_moving = False

    def go_left(self):
        self.hero.direction = Settings.LEFT
        self.hero.is_moving = True
        self.hero.is_hit_wall = False

    def go_right(self):
        self.hero.direction = Settings.RIGHT
        self.hero.is_moving = True
        self.hero.is_hit_wall = False

    def go_up(self):
        self.hero.direction = Settings.UP
        self.hero.is_moving = True
        self.hero.is_hit_wall = False

    def go_down(self):
        self.hero.direction = Settings.DOWN
        self.hero.is_moving = True
        self.hero.is_hit_wall = False

    def pause_(self):
        self.hero.is_moving = False

    def __event_handler(self):
        for event in pygame.event.get():
            # 判断是否是退出游戏
            if event.type == pygame.QUIT:
                TankWar.__game_over()
            elif event.type == pygame.KEYDOWN:
                TankWar.__check_keydown(self, event)
            elif event.type == pygame.KEYUP:
                TankWar.__check_keyup(self, event)

    def __check_collide(self):
        # 保证坦克不移出屏幕
        self.hero.hit_wall()
        for enemy in self.enemies:
            enemy.hit_wall_turn()

        # 子弹击中墙
        for wall in self.walls:
            # 我方英雄子弹击中墙
            for bullet in self.hero.bullets:
                if pygame.sprite.collide_rect(wall, bullet):
                    if wall.type == Settings.RED_WALL:
                        wall.kill()
                        bullet.kill()
                    elif wall.type == Settings.BOSS_WALL:
                        self.game_still = False
                    elif wall.type == Settings.IRON_WALL:
                        bullet.kill()
            # 敌方英雄子弹击中墙
            for enemy in self.enemies:
                for bullet in enemy.bullets:
                    if pygame.sprite.collide_rect(wall, bullet):
                        if wall.type == Settings.RED_WALL:
                            wall.kill()
                            bullet.kill()
                        elif wall.type == Settings.BOSS_WALL:
                            self.game_still = False
                        elif wall.type == Settings.IRON_WALL:
                            bullet.kill()

            # 我方坦克撞墙
            if pygame.sprite.collide_rect(self.hero, wall):
                # 不可穿越墙
                if wall.type == Settings.RED_WALL or wall.type == Settings.IRON_WALL or wall.type == Settings.BOSS_WALL:
                    self.hero.is_hit_wall = True
                    # 移出墙内
                    self.hero.move_out_wall(wall)

            # 敌方坦克撞墙
            for enemy in self.enemies:
                if pygame.sprite.collide_rect(wall, enemy):
                    if wall.type == Settings.RED_WALL or wall.type == Settings.IRON_WALL or wall.type == Settings.BOSS_WALL:
                        enemy.move_out_wall(wall)
                        enemy.random_turn()

        # 子弹击中、敌方坦克碰撞、敌我坦克碰撞
        pygame.sprite.groupcollide(self.hero.bullets, self.enemies, True, True)
        # 敌方子弹击中我方
        for enemy in self.enemies:
            for bullet in enemy.bullets:
                if pygame.sprite.collide_rect(bullet, self.hero):
                    bullet.kill()
                    self.hero.kill()

    def __update_sprites(self):
        if self.hero.is_moving:
            self.hero.update()
        self.walls.update()
        self.hero.bullets.update()
        self.enemies.update()
        for enemy in self.enemies:
            enemy.bullets.update()
            enemy.bullets.draw(self.screen)
        self.enemies.draw(self.screen)
        self.hero.bullets.draw(self.screen)
        self.screen.blit(self.hero.image, self.hero.rect)
        self.walls.draw(self.screen)

    def run_game(self):

        top, right, bottom, left = 90, 360, 285, 580

        num_frames = 0
        gesture = 'pause'
        thresholded = None

        camera = cv2.VideoCapture(0)

        self.__init_game()
        self.__create_sprite()

        while True and self.hero.is_alive and self.game_still:

            self.screen.fill(Settings.SCREEN_COLOR)
            # 1、设置刷新帧率
            self.clock.tick(Settings.FPS)
            # 2、事件监听
            self.__event_handler()

            self.__check_collide()  #碰撞检测

            grabbed, frame = camera.read()
            if not grabbed:
                break

            frame = imutils.resize(frame, width=600)

            frame = cv2.flip(frame, 1)

            clone = frame.copy()

            (height, width) = frame.shape[:2]

            roi = frame[top:bottom, right:left]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            if num_frames < 10:
                self.run_avg(gray, 0.5)
                pass
            else:
                hand = self.segment(gray)

                if hand is not None:
                    (thresholded, segmented) = hand
                    cv2.drawContours(
                        clone, [segmented + (right, top)], -1, (0, 0, 255))
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
            num_frames += 1

            if not thresholded is None:
                gesture, input_im, layout = self.process_gesture(thresholded)
                cv2.imshow('Thesholded', np.vstack([input_im, layout]))
            cv2.imshow('Video Feed', clone)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()



            if gesture == 'left':
                self.go_left()
            if gesture == 'right':
                self.go_right()
            if gesture == 'up':
                self.go_up()
            if gesture == 'down':
                self.go_down()
            if gesture == 'pause':
                # self.pause_()
                self.hero.shot()
            # 3、碰撞监测
            self.__check_collide()
            # 4、更新/绘制精灵/经理组
            self.__update_sprites()
            # 5、更新显示
            pygame.display.update()
        self.__game_over()

    @staticmethod
    def __game_over():
        pygame.quit()
        exit()




if __name__ == '__main__':
    tankWar = TankWar()
    tankWar.run_game()


