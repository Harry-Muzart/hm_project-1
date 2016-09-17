#pygame



import pygame
#import random

## x = pygame.init()
# print(x)

pygame.init()

#RGB 256; 256^3 combs; screen pixels
white = (255,255,255)
black = (0,0,0)
red = (255,0,0)
Blu = (0,0,100)

gameDisplay = pygame.display.set_mode((800,600))

pygame.display.set_caption('ANN')


# pygame.display.flip()
# pygame.display.update()

gameExit = False

lead_x = 300
lead_y = 300

lead_x_change = 0

font = pygame.font.SysFont(None, 25)

#def message_scrn(msg,color):
    #screen_text = font.render(msg, True, color)

#randApple = random.randrange(0, display_width)

#event handling and logic-based
#using while game loop - skeleton B
#

while not gameExit:
    #event handling loop
    for event in pygame.event.get():
        #print(event)
        if event.type == pygame.QUIT:
            gameExit = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                lead_x -= 10
            if event.key == pygame.K_RIGHT:
                lead_x += 10
            if event.key == pygame.K_DOWN:
                lead_y += 10
            if event.key == pygame.K_UP:
                lead_y -= 10
        #if event.type == pygame.MOUSEBUTTONUP:
            #if event.key == pygame.

    if lead_x > 800 or lead_x < 0 or lead_y > 600 or lead_y < 0:
        pygame.draw.rect(gameDisplay, white, [700,10,10,10])
    if lead_x >= 200 and lead_x <= 250 and lead_y >= 200 and lead_y <=250:
        pygame.draw.rect(gameDisplay, white, [700,10,10,10])
        

    gameDisplay.fill(Blu)
    
    pygame.draw.rect(gameDisplay, red, [lead_x,lead_y,10,50])
    pygame.draw.rect(gameDisplay, white, [lead_x,lead_y,10,10])

    gameDisplay.fill(red, rect=[200,200,50,50])

    #minusHealth = gameDisplay.fill(white, rect=[700,10,10,10])
    

    
    pygame.display.update()
            



pygame.quit()

quit()


        

