import os
import pygame
import numpy as np
from dataset import train_data, labels
from ad import Adaline

def button(column,row, margin, txt="", color=(255, 255, 255)):
	pygame.draw.rect(screen, color,
				[(margin + width) * column + margin,
				(margin + height) * row + margin,
				width,
				height])
	label = myfont.render(txt , 1, (0,0,0))
	screen.blit(label, ((margin + width) * column + 2*margin  , (margin + height) * row + margin) )

def up(grid):
	for row in range(1,h):
		grid[row-1] = grid[row]
	grid[h-1] = [0.0] * w
	return grid

def down(grid):
	for row in range(h-2,-1,-1):
		grid[row+1] = grid[row]
	grid[0] = [0.0] * w
	return grid

def right(grid):
	for row in range(h):
		for column in range(w-2,-1,-1):
			grid[row][column+1] = grid[row][column]
		grid[row][0] = 0.0
	return grid

def left(grid):
	for row in range(h):
		for column in range(1,w):
			grid[row][column-1] = grid[row][column]
		grid[row][h-1] = 0.0
	return grid
# initialize the pygame module
pygame.init()

# initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
myfont = pygame.font.SysFont("italic", 25)


# initialize screen 
screen = pygame.display.set_mode((700, 700))
pygame.display.set_caption("Hand written digits predictor")

# width and height of each of cells in the grid
width, height = 20, 20

w = 7
h = 7

# margin of the grid
margin = 5

# grid 
grid = np.zeros((w, h))

net = [Adaline(w*h,0.1,1000) for _ in range(10)]
num = train_data()

for i in range(10):
	net[i].train(num, labels(i))

# main loop
while True:
	screen.fill(pygame.Color("black"))
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			exit(0)
		elif event.type == pygame.MOUSEBUTTONDOWN:
			pos = pygame.mouse.get_pos()
			print('Current possition of the mouse {}'.format(pos))
			column = pos[0] // (width + margin)
			row = pos[1] // (height + margin)
			print('Column={}, Row={}'.format(column, row))
			
			if pos[1] > (height + margin)*7 and pos[0] > margin and pos[1] < (width + margin)*7 + (margin + height) * 2 + margin and (margin + width) * 4 + margin:
				x = pos[0]//(margin + width)
				y = (pos[1]-(height + margin)*7)//(height + margin)
				try:
					grid = num[y*5+x].copy()
				except:
					print('Out of bound')
			elif row > 8 and row < 11 and column < 7:
				if(column == 1 and row == 9):
					grid = up(grid)
				elif(column == 0 and row == 10):
					grid = left(grid)
				elif(column == 1 and row == 10):
					grid = down(grid)
				elif(column == 2 and row == 10):
					grid = right(grid)
				elif(column == 3 and row == 10):
					grid = np.array([[0.0]*w]*h)
				elif(column == 4 and row == 10):
					noise = np.random.binomial(1, 0.1, size=np.shape(grid) )
					grid += noise
					grid = np.where(grid > 0, 1, 0)
			else:
				# set position on grid
				try:
					if grid[row, column] == 0:
						grid[row, column] = 1
					else:
						grid[row, column] = 0
					print('Grid values {}'.format(grid.flatten()))
				except:
					print('Out of bound')

	# interface 
	for row in range(h):
		for column in range(w):
			color = (255, 255, 255)
			if grid[row, column] == 1:
				color = (0, 255, 0)
			button(column,row, margin, color=color)

	# button
	for row in range(2):
		for column in range(5):
			button(column,row+7, margin, txt=str(5*row+column) )

	button(0,10 , margin, txt="<" )
	button(1,10 , margin, txt="v" )
	button(2,10 , margin, txt=">" )
	button(1,9 , margin, txt="^" )
	button(3,10 , margin, txt="X" )
	button(4,10 , margin, txt="N" )
	result = []
	for i in range(10):
		result.append(net[i].predict(grid.flatten()))
	for i in range(10):
		text = "adaline {} : {}".format(str(i), result[i] )
		if result[i] == max(result):
			colorV = (0,255,0)
		else:
			colorV = (255,255,255)
		label = myfont.render(text, 1, colorV)
		screen.blit(label, (h * (width + margin) + w * margin, margin + (margin * i) + w*h*i ))		
			
	pygame.display.flip()

pygame.quit()




