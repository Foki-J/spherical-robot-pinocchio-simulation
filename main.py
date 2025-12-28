import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from pin_dynamics import RobotDynamics
from visual import RobotVisualizer


class DataLogger:
    """Keep a sliding window of scalar samples for plotting."""

    def __init__(self, max_points: int = 300) -> None:
        self.max_points = max_points
        self.data: Dict[str, List[float]] = {}
        self.colors = {
            "desired": (0, 255, 0),
            "actual": (255, 0, 0),
            "output": (0, 128, 255),
        }

    def add_data(self, key: str, value: float) -> None:
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(float(value))
        if len(self.data[key]) > self.max_points:
            self.data[key].pop(0)

    def clear(self) -> None:
        self.data.clear()


def draw_curve_plot(
    screen: pygame.Surface,
    data_logger: DataLogger,
    desired_value: float,
    actual_value: float,
    output_value: float,
    position: Tuple[int, int] = (20, 20),
    size: Tuple[int, int] = (600, 400),
    title: str = "Control",
    y_range: Tuple[float, float] = (-10.0, 10.0),
) -> None:
    """Render a simple history plot for desired/actual/output signals."""

    data_logger.add_data("desired", desired_value)
    data_logger.add_data("actual", actual_value)
    data_logger.add_data("output", output_value)

    bg_color = (30, 30, 40)
    grid_color = (60, 60, 70)
    border_color = (100, 100, 120)
    title_color = (220, 220, 220)
    font = pygame.font.SysFont("Arial", 16)
    small_font = pygame.font.SysFont("Arial", 12)

    x0, y0 = position
    width, height = size
    y_min, y_max = y_range

    pygame.draw.rect(screen, bg_color, (x0, y0, width, height))
    pygame.draw.rect(screen, border_color, (x0, y0, width, height), 2)

    title_surface = font.render(title, True, title_color)
    screen.blit(title_surface, (x0 + 10, y0 + 5))

    for i in range(5):
        y_pos = y0 + height - (i + 1) * height / 5
        pygame.draw.line(screen, grid_color, (x0, y_pos), (x0 + width, y_pos), 1)

    for i in range(10):
        x_pos = x0 + (i + 1) * width / 10
        pygame.draw.line(screen, grid_color, (x_pos, y0), (x_pos, y0 + height), 1)

    for i in range(5):
        value = y_min + (y_max - y_min) * i / 4
        y_pos = y0 + height - (i + 1) * height / 5
        label = small_font.render(f"{value:.1f}", True, title_color)
        screen.blit(label, (x0 + 5, y_pos - 8))

    for key, color in data_logger.colors.items():
        data = data_logger.data.get(key)
        if not data or len(data) < 2:
            continue
        points: List[Tuple[float, float]] = []
        for i, value in enumerate(data):
            x_pos = x0 + width - (len(data) - i) * width / len(data)
            normalized = (value - y_min) / (y_max - y_min)
            y_pos = y0 + height - normalized * height
            points.append((x_pos, y_pos))
        pygame.draw.lines(screen, color, False, points, 2)

    legend_y = y0 + height - 60
    for i, (key, color) in enumerate(data_logger.colors.items()):
        if key in data_logger.data:
            pygame.draw.rect(screen, color, (x0 + 10, legend_y + i * 20, 12, 12))
            label = small_font.render(key.capitalize(), True, title_color)
            screen.blit(label, (x0 + 28, legend_y + i * 20))

    info_y = y0 + 30
    display_values = [
        ("desired", desired_value),
        ("actual", actual_value),
        ("output", output_value),
    ]
    for i, (key, value) in enumerate(display_values):
        color = data_logger.colors[key]
        text = small_font.render(f"{key.title()}: {value:.3f}", True, color)
        screen.blit(text, (x0 + width - 150, info_y + i * 18))


def get_remote_input(joystick: Optional[pygame.joystick.Joystick]) -> Tuple[float, float]:
    axis_x = 0.0
    axis_y = 0.0

    if joystick:
        axis_x = -joystick.get_axis(1)
        axis_y = -joystick.get_axis(2)
    else:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            axis_x += 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            axis_x -= 1.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            axis_y += 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            axis_y -= 1.0

    first_velocity_hope = axis_x * 6.0
    roll_hope = -axis_y * 0.5
    return first_velocity_hope, roll_hope


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float, max_integral: float = 10.0) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.max_integral = max_integral

    def compute(self, error: float, derivative: float = 0.0) -> float:
        self.integral += error * self.dt
        self.integral = max(-self.max_integral, min(self.integral, self.max_integral))
        return self.kp * error + self.ki * self.integral + self.kd * derivative


roll_pid: Optional[PIDController] = None
first_velocity_pid: Optional[PIDController] = None
w_pid: Optional[PIDController] = None


def controller(sim: RobotDynamics, first_velocity_hope: float, roll_hope: float) -> Tuple[np.ndarray, float, float, float]:
    motors_velocity, motors_position, euler_angles, angular_velocity = sim.get_state()
    tau_actuators = np.zeros(3)
    assert roll_pid and first_velocity_pid and w_pid
    tau_actuators[1] = roll_pid.compute(roll_hope - euler_angles[0], angular_velocity[0])
    tau_actuators[0] = first_velocity_pid.compute(first_velocity_hope - motors_velocity[0])
    tau_actuators[2] = w_pid.compute(-angular_velocity[0])
    return tau_actuators, euler_angles[0], angular_velocity[0], motors_velocity[0]


def main() -> None:
    pygame.init()
    pygame.joystick.init()
    time_step = 0.005
    render_stride = 5

    use_joystick = pygame.joystick.get_count() > 0
    if use_joystick:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Joystick connected: {joystick.get_name()}")
    else:
        joystick = None
        print("No joystick detected. Use arrow/WASD keys for input.")

    global roll_pid, first_velocity_pid, w_pid
    roll_pid = PIDController(kp=-40.0, ki=-0.1, kd=-20.0, dt=time_step)
    first_velocity_pid = PIDController(kp=20.0, ki=0.1, kd=0.2, dt=time_step)
    w_pid = PIDController(kp=50.0, ki=0.1, kd=0.2, dt=time_step)

    urdf_path = "sr_description/urdf/ball.urdf"
    mesh_dir = "sr_description"

    sim = RobotDynamics(urdf_path=urdf_path, shell_radius=0.4, dt=time_step)
    visualizer = RobotVisualizer(urdf_path=urdf_path, mesh_dir=mesh_dir)

    screen = pygame.display.set_mode((800, 500))
    pygame.display.set_caption("Spherical Robot Simulation")

    data_logger = DataLogger(max_points=300)

    clock = pygame.time.Clock()
    running = True
    last_time = time.time()
    loop_counter = 0

    print("Simulation started...")
    print("Plot legend: green=desired, red=actual, blue=output")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        first_velocity_hope, roll_hope = get_remote_input(joystick)

        tau_actuators, actual_roll, actual_angular_velocity, actual_velocity = controller(
            sim, first_velocity_hope, roll_hope
        )

        screen.fill((20, 20, 30))

        draw_curve_plot(
            screen=screen,
            data_logger=data_logger,
            desired_value=first_velocity_hope / 10.0,
            actual_value=actual_velocity / 10.0,
            output_value=tau_actuators[0] / 50.0,
            position=(50, 50),
            size=(700, 350),
            title="Wheel Velocity Control (scaled)",
            y_range=(-1.0, 1.0),
        )

        sim.step_forward_dynamics_manual(tau_actuators)
        if loop_counter % render_stride == 0:
            visualizer.display(sim.q.copy())

        pygame.display.flip()

        dt = time.time() - last_time
        sleep_time = max(0.0, time_step - dt)
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_time = time.time()
        clock.tick(240)
        loop_counter += 1

    pygame.quit()


if __name__ == "__main__":
    main()
