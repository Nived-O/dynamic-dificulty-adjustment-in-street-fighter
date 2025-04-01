import pygame, os
import config
import menu
from game import Point

if __name__ == "__main__":
    print('loadinaaag...')
    options = config.OptionConfig()
    
    pygame.init()
    pygame.mixer.init()
    config.Screen()
    
    screen = pygame.display.set_mode((650, 500), 0, 32)
    #gscreen = pygame.Surface((650, 500), 0, 32)

    menu = menu.MainMenu(screen, Point(20,10))
    menu.mainloop()
    