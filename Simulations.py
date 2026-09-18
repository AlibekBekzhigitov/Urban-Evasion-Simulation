import math
import random
import pygame
from PIL import Image

# Initialization
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "Qoyanshyq Engine: False Positives & POMDP Noise Model"
)
clock = pygame.time.Clock()

# Color Palette
COLOR_BG = (220, 220, 220)
COLOR_BUILDING = (70, 75, 85)
COLOR_ROAD = (110, 110, 110)
COLOR_PARKING = (150, 150, 150)
COLOR_LINE = (240, 240, 240)
COLOR_SIDEWALK = (180, 175, 165)
COLOR_COURTYARD = (160, 190, 160)

COLOR_CIVILIAN = (210, 170, 40)
COLOR_EVADER = (0, 120, 255)
COLOR_CATCHER = (235, 30, 30)
COLOR_CAR = (80, 100, 120)
COLOR_MINIVAN = (30, 30, 40)

# Topology Layers
Y_ROAD = 80
Y_PARKING = 180
Y_SIDEWALK = 250
Y_BUILDINGS = 330

ARCH_X_MIN, ARCH_X_MAX = 480, 580


class ParkingSlot:

  def __init__(self, x, y, width=45, height=60):
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    self.occupied = False


class Agent:

  def __init__(self, x, y, is_evader=False):
    self.x = x
    self.y = y
    self.is_evader = is_evader
    self.speed = random.uniform(0.2, 0.35)
    self.sprint_speed = 0.85
    self.captured = False
    self.in_van = False
    self.state = "WALKING"

  def move(self):
    if not self.captured and not self.in_van and self.state == "WALKING":
      self.x += self.speed
      self.y += random.uniform(-0.02, 0.02)

  def draw(self, surface):
    if self.in_van:
      return
    color = COLOR_EVADER if self.is_evader else COLOR_CIVILIAN
    radius = 6 if self.is_evader else 4
    pygame.draw.circle(surface, color, (int(self.x), int(self.y)), radius)


class Vehicle:

  def __init__(self, x, y, is_minivan=False, is_operational=False):
    self.x = x
    self.y = y
    self.is_minivan = is_minivan
    self.is_operational = is_operational  # True — Patrol Van, False — Civilian
    self.state = "CRUISING"
    self.speed = random.uniform(1.6, 2.3)
    self.target_slot = None
    self.park_time = 0
    self.stop_count = 0
    self.deployed_for_current_stop = False
    self.random_deploy_delay = random.randint(500, 1600)

  def update(self, parking_slots):
    if self.state == "CRUISING":
      self.y = Y_ROAD + 25
      self.x += self.speed

      # Both civilian and operational minivans can park
      if self.is_minivan and self.x > random.randint(100, 350):
        # Patrol has its own timing, civilian minivans park with a random probability
        should_park = (self.is_operational and self.stop_count == 0) or (
            not self.is_operational and random.random() < 0.015
        )
        if should_park:
          free_slots = [
              s for s in parking_slots if not s.occupied and s.x > self.x + 30
          ]
          if free_slots:
            self.target_slot = free_slots[0]
            self.target_slot.occupied = True
            self.state = "MANEUVER_PARK"
            if self.is_operational:
              self.stop_count = 1

    elif self.state == "MANEUVER_PARK":
      target_x = self.target_slot.x + 5
      target_y = self.target_slot.y + 5
      dx = target_x - self.x
      dy = target_y - self.y
      dist = math.hypot(dx, dy)
      if dist > 2:
        self.x += (dx / dist) * 1.3
        self.y += (dy / dist) * 1.3
      else:
        self.x = target_x
        self.y = target_y
        self.state = "PARKED"
        self.park_time = pygame.time.get_ticks()

    elif self.state == "RELOCATING":
      if self.target_slot:
        self.target_slot.occupied = False
        self.target_slot = None

      target_y = Y_ROAD + 25
      if abs(self.y - target_y) > 2:
        self.y -= 1.0
        self.x += 1.0
      else:
        self.y = target_y
        self.state = "CRUISING_TO_NEXT"

    elif self.state == "CRUISING_TO_NEXT":
      self.y = Y_ROAD + 25
      self.x += self.speed

      if self.is_operational and self.stop_count == 1 and self.x > 480:
        free_slots = [
            s for s in parking_slots if not s.occupied and s.x > self.x + 30
        ]
        if free_slots and random.random() < 0.06:
          self.target_slot = free_slots[0]
          self.target_slot.occupied = True
          self.state = "MANEUVER_PARK"
          self.stop_count = 2
          self.deployed_for_current_stop = False
          self.random_deploy_delay = random.randint(400, 1200)
        elif self.x > WIDTH - 100:
          self.state = "LEAVING"

    elif self.state == "LEAVING":
      if self.target_slot:
        self.target_slot.occupied = False
        self.target_slot = None
      target_y = Y_ROAD + 25
      if abs(self.y - target_y) > 2:
        self.y -= 1.0
        self.x += 1.0
      else:
        self.y = target_y
        self.x += self.speed

  def draw(self, surface):
    color = COLOR_MINIVAN if self.is_minivan else COLOR_CAR
    w, h = (38, 50) if self.is_minivan else (30, 45)
    pygame.draw.rect(
        surface, color, (int(self.x), int(self.y), w, h), border_radius=3
    )


class Catcher:

  def __init__(self, x, y, home_minivan):
    self.x = x
    self.y = y
    self.speed = 0.65
    self.target = None
    self.home_minivan = home_minivan
    self.inside_minivan = False

  def update(self, pedestrians, stop_quota_reached):
    if self.inside_minivan:
      return

    if self.target and self.target.captured:
      dx = self.home_minivan.x + 15 - self.x
      dy = self.home_minivan.y + 25 - self.y
      dist = math.hypot(dx, dy)

      self.target.x = self.x
      self.target.y = self.y

      if dist > 6:
        self.x += (dx / dist) * self.speed
        self.y += (dy / dist) * self.speed
      else:
        self.inside_minivan = True
        self.target.in_van = True
        self.target = None
      return

    if stop_quota_reached:
      dx = self.home_minivan.x + 15 - self.x
      dy = self.home_minivan.y + 25 - self.y
      dist = math.hypot(dx, dy)
      if dist > 6:
        self.x += (dx / dist) * self.speed
        self.y += (dy / dist) * self.speed
      else:
        self.inside_minivan = True
      return

    # Search for targets strictly within the parking slot radius (55px)
    if not self.target:
      valid = [
          p
          for p in pedestrians
          if not p.captured
          and not p.in_van
          and abs(p.x - self.home_minivan.x) <= 55
          and p.y < Y_BUILDINGS + 10
      ]
      if valid:
        self.target = min(
            valid, key=lambda p: math.hypot(p.x - self.x, p.y - self.y)
        )

    if self.target and not self.target.captured:
      angle = math.atan2(self.target.y - self.y, self.target.x - self.x)
      self.x += math.cos(angle) * self.speed
      self.y += math.sin(angle) * self.speed
      if math.hypot(self.target.x - self.x, self.target.y - self.y) < 6:
        self.target.captured = True

  def draw(self, surface):
    if not self.inside_minivan:
      pygame.draw.circle(surface, COLOR_CATCHER, (int(self.x), int(self.y)), 4)


def init_world():
  parking_slots = [ParkingSlot(80 + i * 65, Y_PARKING + 5) for i in range(16)]
  parked_cars = []

  for slot in parking_slots:
    if random.random() < 0.4:
      slot.occupied = True
      v = Vehicle(slot.x + 5, slot.y + 5, is_minivan=False)
      v.state = "PARKED"
      v.speed = 0
      parked_cars.append(v)

  pedestrians = [
      Agent(
          random.randint(-150, 650),
          random.randint(Y_SIDEWALK + 10, Y_SIDEWALK + 55),
      )
      for _ in range(35)
  ]

  # Evader spawns off-screen at the start along with traffic
  evader = Agent(-30, Y_SIDEWALK + 30, is_evader=True)
  pedestrians.append(evader)

  total_kpi_quota = random.randint(3, 4)
  op_minivan_order = random.randint(3, 7)

  return (
      parking_slots,
      parked_cars,
      pedestrians,
      evader,
      [],
      [],
      total_kpi_quota,
      op_minivan_order,
  )


def run_simulation():
  (
      parking_slots,
      parked_cars,
      pedestrians,
      evader,
      traffic,
      catchers,
      total_kpi_quota,
      op_minivan_order,
  ) = init_world()

  paused = False
  recording = False
  frames = []
  font = pygame.font.SysFont("Arial", 16)
  stop1_target = 2
  spawned_vehicles_count = 0
  op_minivan_ref = None

  running = True
  while running:
    current_time = pygame.time.get_ticks()

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_SPACE:
          paused = not paused
        elif event.key == pygame.K_r:
          (
              parking_slots,
              parked_cars,
              pedestrians,
              evader,
              traffic,
              catchers,
              total_kpi_quota,
              op_minivan_order,
          ) = init_world()
          spawned_vehicles_count = 0
          op_minivan_ref = None
        elif event.key == pygame.K_g:
          recording = not recording

    if not paused:
      screen.fill(COLOR_BG)

      # 1. TOPOLOGY
      pygame.draw.rect(screen, COLOR_ROAD, (0, Y_ROAD, WIDTH, 100))
      pygame.draw.rect(screen, COLOR_PARKING, (0, Y_PARKING, WIDTH, 70))
      for slot in parking_slots:
        pygame.draw.rect(
            screen, COLOR_LINE, (slot.x, slot.y, slot.width, slot.height), 2
        )
      pygame.draw.rect(screen, COLOR_SIDEWALK, (0, Y_SIDEWALK, WIDTH, 80))
      pygame.draw.rect(
          screen, COLOR_COURTYARD, (0, Y_BUILDINGS, WIDTH, HEIGHT - Y_BUILDINGS)
      )

      pygame.draw.rect(screen, COLOR_BUILDING, (40, Y_BUILDINGS + 40, 420, 160))
      pygame.draw.rect(
          screen, COLOR_BUILDING, (580, Y_BUILDINGS + 40, 580, 160)
      )
      pygame.draw.rect(
          screen, COLOR_COURTYARD, (ARCH_X_MIN, Y_BUILDINGS, 100, 200)
      )

      # 2. MARKOV TRAFFIC GENERATOR
      if random.random() < 0.015:
        spawned_vehicles_count += 1
        if spawned_vehicles_count == op_minivan_order and op_minivan_ref is None:
          # Spawn OPERATIONAL MINIVAN
          op_minivan_ref = Vehicle(
              -80, Y_ROAD + 25, is_minivan=True, is_operational=True
          )
          traffic.append(op_minivan_ref)
        else:
          # Civilian vehicles (including civilian minivans)
          is_mv = random.random() < 0.25
          traffic.append(
              Vehicle(-60, Y_ROAD + 25, is_minivan=is_mv, is_operational=False)
          )

      # 3. VEHICLE PROCESSING
      all_vehicles = traffic + parked_cars
      total_captured = sum(1 for p in pedestrians if p.in_van)

      for v in all_vehicles:
        v.update(parking_slots)
        v.draw(screen)

      # Deployment only occurs from the operational patrol van
      if (
          op_minivan_ref
          and op_minivan_ref.state == "PARKED"
          and not op_minivan_ref.deployed_for_current_stop
      ):
        if (
            current_time - op_minivan_ref.park_time
            > op_minivan_ref.random_deploy_delay
        ):
          op_minivan_ref.deployed_for_current_stop = True
          catchers.clear()

          # Direct intercept at doors within slot radius (<=55px)
          if (
              abs(op_minivan_ref.x - evader.x) <= 55
              and not evader.captured
              and not evader.in_van
          ):
            evader.captured = True

          for i in range(3):
            c = Catcher(
                op_minivan_ref.x + 5 + (i * 8), Y_SIDEWALK - 5, op_minivan_ref
            )
            if evader.captured and not evader.in_van and i == 0:
              c.target = evader
            catchers.append(c)

      # Relocation between stops
      if (
          op_minivan_ref
          and op_minivan_ref.state == "PARKED"
          and op_minivan_ref.deployed_for_current_stop
          and catchers
      ):
        if all(c.inside_minivan for c in catchers):
          if (
              total_captured < total_kpi_quota
              and op_minivan_ref.stop_count == 1
          ):
            op_minivan_ref.state = "RELOCATING"
          else:
            op_minivan_ref.state = "LEAVING"

      # 4. PEDESTRIAN SPAWN
      if random.random() < 0.02:
        pedestrians.append(
            Agent(-20, random.randint(Y_SIDEWALK + 10, Y_SIDEWALK + 55))
        )

      # 5. EVADER DYNAMICS (False Positive & Reaction to Ahead Maneuvers)
      # Evader detects ANY minivan parked ahead (Civilian or Operational)
      parked_minivans_ahead = [
          v
          for v in all_vehicles
          if v.is_minivan
          and v.state in ["MANEUVER_PARK", "PARKED"]
          and v.x > evader.x - 10
          and v.x < evader.x + 220
      ]

      active_catchers = [c for c in catchers if not c.inside_minivan]

      # Escape behavior triggers on ACTIVE PATROL DEPLOYMENT OR SUSPICIOUS MINIVAN PARKING AHEAD
      if (
          (active_catchers or parked_minivans_ahead)
          and not evader.captured
          and not evader.in_van
      ):
        evader.state = "ESCAPING"
        arch_center_x = (ARCH_X_MIN + ARCH_X_MAX) // 2

        if abs(evader.x - arch_center_x) > 10 and evader.y < Y_BUILDINGS + 15:
          angle = math.atan2(
              Y_SIDEWALK + 40 - evader.y, arch_center_x - evader.x
          )
        else:
          angle = math.atan2((Y_BUILDINGS + 150) - evader.y, 0)

        evader.x += math.cos(angle) * evader.sprint_speed
        evader.y += math.sin(angle) * evader.sprint_speed
      else:
        evader.move()

      # 6. DRAWING AND UPDATING
      for p in pedestrians:
        if p != evader:
          p.move()
        p.draw(screen)

      evader.draw(screen)

      stop_quota_reached = (
          (total_captured >= stop1_target)
          if (op_minivan_ref and op_minivan_ref.stop_count == 1)
          else (total_captured >= total_kpi_quota)
      )

      for c in catchers:
        c.update(pedestrians, stop_quota_reached)
        c.draw(screen)

      txt = font.render(
          f"In Van: {total_captured}/{total_kpi_quota} | Evader:"
          f" {'IN_VAN' if evader.in_van else evader.state} | [SPACE]: Pause |"
          " [R]: Reset",
          True,
          (10, 10, 10),
      )
      screen.blit(txt, (10, 10))

      if recording:
        image_string = pygame.image.tostring(screen, "RGB")
        pil_image = Image.frombytes("RGB", (WIDTH, HEIGHT), image_string)
        pil_image = pil_image.resize((600, 400), Image.Resampling.LANCZOS)
        frames.append(pil_image)

      pygame.display.flip()

    clock.tick(60)

  if frames:
    frames[0].save(
        "simulation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )

  pygame.quit()


if __name__ == "__main__":
  run_simulation()