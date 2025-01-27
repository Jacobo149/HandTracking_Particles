import cv2
import mediapipe as mp
import pygame
import random
import math

# Initialize pygame
pygame.init()

# Window size
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Particle System based on Hand Gesture")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Define a Particle class
class Particle:
    def __init__(self, x, y, velocity):
        # Ensure x and y are valid integers
        self.x = int(x)
        self.y = int(y)
        self.velocity = velocity
        self.size = random.randint(2, 4)
        self.color = (255, random.randint(0, 255), random.randint(0, 255))

    def move(self):
        self.x += self.velocity[0]
        self.y += self.velocity[1]

        # Bounce off the walls
        if self.x <= 0 or self.x >= WIDTH:
            self.velocity[0] = -self.velocity[0]
        if self.y <= 0 or self.y >= HEIGHT:
            self.velocity[1] = -self.velocity[1]

    def draw(self):
        # Ensure x and y are valid numbers (float or int) before drawing
        if isinstance(self.x, (int, float)) and isinstance(self.y, (int, float)):
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)
        else:
            print(f"Invalid particle position: ({self.x}, {self.y})")

def recognize_gesture(landmarks):
    """
    Recognize basic gestures based on hand landmarks and return a boolean value.
    """
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Gesture: Open Hand (fingers spread apart)
    if index_tip.x > thumb_tip.x and middle_tip.x > index_tip.x and ring_tip.x > middle_tip.x and pinky_tip.x > ring_tip.x:
        return "Open Hand"

    # Gesture: Fist (fingers curled in)
    if abs(index_tip.x - thumb_tip.x) < 0.02 and abs(middle_tip.x - index_tip.x) < 0.02 and abs(ring_tip.x - middle_tip.x) < 0.02 and abs(pinky_tip.x - ring_tip.x) < 0.02:
        return "Fist"

    return "Unknown"

def get_direction_to_origin(x, y):
    """Calculate the direction vector towards the origin (center of the screen)."""
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    direction_x = center_x - x
    direction_y = center_y - y
    length = math.sqrt(direction_x ** 2 + direction_y ** 2)

    # Normalize the direction vector to avoid large velocities
    if length != 0:
        direction_x /= length
        direction_y /= length

    return direction_x, direction_y

# Initialize particles list
particles = []

# Main loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Process the image for hand tracking
    image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Detect the hand gesture
    gesture = "Unknown"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture = recognize_gesture(hand_landmarks.landmark)

    # Add new particles (randomly distributed in the window)
    for _ in range(5):  # Adjust particle spawn rate here
        particles.append(Particle(random.randint(0, WIDTH), random.randint(0, HEIGHT), [random.uniform(-1, 1), random.uniform(-1, 1)]))

    # Update particle behavior based on the gesture
    for particle in particles:
        if gesture == "Open Hand":
            # Open hand: Disperse particles outwards
            direction_x, direction_y = get_direction_to_origin(particle.x, particle.y)
            particle.velocity = [direction_x * 2, direction_y * 2]
        elif gesture == "Fist":
            # Fist: Converge particles towards the origin (center)
            direction_x, direction_y = get_direction_to_origin(particle.x, particle.y)
            particle.velocity = [direction_x * -2, direction_y * -2]

        particle.move()

    # Clear screen and redraw particles
    screen.fill((0, 0, 0))  # Clear the screen with black
    for particle in particles:
        particle.draw()

    # Handle window events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()

    # Show the window
    pygame.display.flip()

    # Show hand tracking in a separate window (optional)
    cv2.imshow("Hand Tracking", image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
pygame.quit()
