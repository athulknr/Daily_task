import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the window
width, height = 800, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption('Bouncing Ball Animation')

# Define colors
black = (0, 0, 0)
light_red = (255, 102, 102)  # RGB for light red

# Ball settings
ball_radius = 20
ball_pos = [width // 2, 0]  # Start from the top-center of the window
ball_vel = [0, 0]  # Initial velocity

# Constants
gravity = 0.5  # Gravitational acceleration
elasticity = 0.8  # Elasticity factor

# Clock object to control the frame rate
clock = pygame.time.Clock()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Apply gravity
    ball_vel[1] += gravity

    # Move the ball
    ball_pos[0] += ball_vel[0]
    ball_pos[1] += ball_vel[1]

    # Bounce the ball off the ground
    if ball_pos[1] + ball_radius > height:
        ball_pos[1] = height - ball_radius
        ball_vel[1] = -ball_vel[1] * elasticity

    # Clear the screen
    window.fill(black)

    # Draw the ball
    pygame.draw.circle(window, light_red, (int(ball_pos[0]), int(ball_pos[1])), ball_radius)

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
