import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from mpc import PlanarMPC
from path_manager import PathManager
from pin_dynamics import RobotDynamics
from state_estimator import PlanarStateEstimator
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
    title: str = "PID Control",
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


MAP_RECT = pygame.Rect(820, 60, 260, 260)
MAP_BOUNDS = np.array([[-3.0, 3.0], [-3.0, 3.0]], dtype=float)
MAP_AXIS_ORDER = (1, 0)  # swap axes: world Y -> panel X, world X -> panel Y
MAP_AXIS_SIGN = np.array([1.0, 1.0], dtype=float)


def world_to_panel(world_xy: np.ndarray) -> np.ndarray:
    return np.array(
        [
            MAP_AXIS_SIGN[0] * world_xy[MAP_AXIS_ORDER[0]],
            MAP_AXIS_SIGN[1] * world_xy[MAP_AXIS_ORDER[1]],
        ],
        dtype=float,
    )


def panel_to_world(panel_xy: np.ndarray) -> np.ndarray:
    world = np.zeros(2, dtype=float)
    world[MAP_AXIS_ORDER[0]] = MAP_AXIS_SIGN[0] * panel_xy[0]
    world[MAP_AXIS_ORDER[1]] = MAP_AXIS_SIGN[1] * panel_xy[1]
    return world


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def world_to_map_pixel(world_xy: np.ndarray) -> Tuple[int, int]:
    panel_xy = world_to_panel(world_xy)
    x_min, x_max = MAP_BOUNDS[0]
    y_min, y_max = MAP_BOUNDS[1]
    x_norm = _clamp01((panel_xy[0] - x_min) / (x_max - x_min))
    y_norm = _clamp01((panel_xy[1] - y_min) / (y_max - y_min))
    x_pix = MAP_RECT.left + x_norm * MAP_RECT.width
    y_pix = MAP_RECT.bottom - y_norm * MAP_RECT.height
    return int(x_pix), int(y_pix)


def map_pixel_to_world(pixel_xy: Tuple[int, int]) -> np.ndarray:
    x_min, x_max = MAP_BOUNDS[0]
    y_min, y_max = MAP_BOUNDS[1]
    x_norm = _clamp01((pixel_xy[0] - MAP_RECT.left) / MAP_RECT.width)
    y_norm = _clamp01((MAP_RECT.bottom - pixel_xy[1]) / MAP_RECT.height)
    panel_x = x_min + x_norm * (x_max - x_min)
    panel_y = y_min + y_norm * (y_max - y_min)
    return panel_to_world(np.array([panel_x, panel_y], dtype=float))


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def draw_path_panel(
    screen: pygame.Surface,
    path_manager: PathManager,
    robot_xy: np.ndarray,
    path_following: bool,
    cursor_world: Optional[np.ndarray] = None,
    predicted_states: Optional[np.ndarray] = None,
) -> None:
    bg_color = (28, 32, 38)
    border_color = (90, 95, 110)
    grid_color = (55, 60, 70)
    raw_color = (220, 180, 60)
    smooth_color = (0, 180, 220)
    predict_color = (100, 220, 140)
    robot_color = (235, 70, 70)
    cursor_color = (120, 200, 120)

    pygame.draw.rect(screen, bg_color, MAP_RECT)
    pygame.draw.rect(screen, border_color, MAP_RECT, 2)

    font_small = pygame.font.SysFont("Arial", 14)

    x_min, x_max = MAP_BOUNDS[0]
    y_min, y_max = MAP_BOUNDS[1]
    for x in range(math.ceil(x_min), math.floor(x_max) + 1):
        x_norm = (x - x_min) / (x_max - x_min)
        x_pix = MAP_RECT.left + x_norm * MAP_RECT.width
        pygame.draw.line(screen, grid_color, (x_pix, MAP_RECT.top), (x_pix, MAP_RECT.bottom), 1)
        label = font_small.render(f"{x}", True, (140, 140, 150))
        screen.blit(label, (x_pix - 10, MAP_RECT.bottom + 4))

    for y in range(math.ceil(y_min), math.floor(y_max) + 1):
        y_norm = (y - y_min) / (y_max - y_min)
        y_pix = MAP_RECT.bottom - y_norm * MAP_RECT.height
        pygame.draw.line(screen, grid_color, (MAP_RECT.left, y_pix), (MAP_RECT.right, y_pix), 1)

    raw_points = path_manager.raw_points()
    if raw_points:
        raw_pixels = [world_to_map_pixel(p) for p in raw_points]
        for pixel in raw_pixels:
            pygame.draw.circle(screen, raw_color, pixel, 3)
        if len(raw_pixels) > 1:
            pygame.draw.lines(screen, raw_color, False, raw_pixels, 1)

    samples = path_manager.get_path_samples()
    if len(samples) > 1:
        smooth_pixels = [world_to_map_pixel(sample.position) for sample in samples]
        pygame.draw.lines(screen, smooth_color, False, smooth_pixels, 2)

    if predicted_states is not None and len(predicted_states) > 0:
        predicted_pixels = [world_to_map_pixel(state[:2]) for state in predicted_states]
        if len(predicted_pixels) > 1:
            pygame.draw.lines(screen, predict_color, False, predicted_pixels, 1)
        for pixel in predicted_pixels[:5]:
            pygame.draw.circle(screen, predict_color, pixel, 2)

    robot_pixel = world_to_map_pixel(robot_xy)
    pygame.draw.circle(screen, robot_color, robot_pixel, 6)
    pygame.draw.circle(screen, (0, 0, 0), robot_pixel, 6, 1)

    if cursor_world is not None:
        cursor_pixel = world_to_map_pixel(cursor_world)
        pygame.draw.circle(screen, cursor_color, cursor_pixel, 4, 1)

    status_text = "FOLLOW" if path_following else "EDIT"
    status_color = (0, 200, 180) if path_following else (200, 200, 180)
    status_surface = font_small.render(f"Mode: {status_text}", True, status_color)
    screen.blit(status_surface, (MAP_RECT.left, MAP_RECT.top - 24))

    info_lines = [
        f"Waypoints: {len(raw_points)}",
        "LMB drag/add path",
        "RMB undo last point",
        "Enter build path | C clear",
        "Space toggle follow",
    ]
    for i, line in enumerate(info_lines):
        surface = font_small.render(line, True, (200, 200, 200))
        screen.blit(surface, (MAP_RECT.left, MAP_RECT.bottom + 20 + i * 18))


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
second_position_pid: Optional[PIDController] = None
w_pid: Optional[PIDController] = None


def controller(sim: RobotDynamics, first_velocity_hope: float, roll_hope: float) -> Tuple[np.ndarray, float, float, float]:
    motors_velocity, motors_position, euler_angles, angular_velocity = sim.get_state()
    tau_actuators = np.zeros(3)
    assert roll_pid and first_velocity_pid and w_pid
    tau_actuators[1] = roll_pid.compute(roll_hope - euler_angles[0], angular_velocity[0])
    tau_actuators[0] = first_velocity_pid.compute(first_velocity_hope - motors_velocity[0])
    tau_actuators[2] = w_pid.compute(-angular_velocity[0])
    return tau_actuators, euler_angles[0], angular_velocity[0], motors_velocity[0]


def build_planar_reference(
    planar_state,
    path_manager: PathManager,
    horizon: int,
    dt: float,
    target_speed: float = 0.4,
) -> List[np.ndarray]:
    references = [planar_state.as_vector()]
    if not path_manager.has_path():
        references.extend([planar_state.as_vector()] * horizon)
        return references

    origin = planar_state.position
    for k in range(1, horizon + 1):
        lookahead = target_speed * dt * k
        sample = path_manager.sample_along_path(origin, lookahead)
        if sample is None:
            references.append(references[-1])
            continue
        pos = sample.position
        vel = sample.tangent * target_speed
        references.append(np.array([pos[0], pos[1], vel[0], vel[1]], dtype=float))
    return references


def compute_autonomous_command(
    planar_state,
    mpc_solution,
    references,
    shell_radius: float,
    mpc_dt: float,
    yaw_gain: float = 0.8,
    yaw_damping: float = 0.35,
) -> Tuple[float, float]:
    desired_state = references[1]
    desired_velocity = desired_state[2:]

    forward = np.array([math.cos(planar_state.yaw), math.sin(planar_state.yaw)])
    lateral = np.array([-math.sin(planar_state.yaw), math.cos(planar_state.yaw)])

    forward_acc = float(np.dot(mpc_solution.control, forward))
    lateral_acc = float(np.dot(mpc_solution.control, lateral))
    forward_velocity_target = float(np.dot(desired_velocity, forward))

    velocity_cmd = forward_velocity_target + 0.25 * forward_acc * mpc_dt
    velocity_cmd = float(np.clip(velocity_cmd, -2.0, 2.0))
    first_velocity_hope = velocity_cmd / max(shell_radius, 1e-4)

    desired_heading = math.atan2(desired_velocity[1], desired_velocity[0])
    yaw_error = wrap_to_pi(desired_heading - planar_state.yaw)
    roll_cmd = lateral_acc / 9.81 + yaw_gain * yaw_error - yaw_damping * planar_state.yaw_rate
    roll_cmd = float(np.clip(roll_cmd, -0.4, 0.4))

    return first_velocity_hope, roll_cmd


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

    global roll_pid, first_velocity_pid, second_position_pid, w_pid
    roll_pid = PIDController(kp=-40.0, ki=-0.1, kd=-20.0, dt=time_step)
    first_velocity_pid = PIDController(kp=20.0, ki=0.1, kd=0.2, dt=time_step)
    second_position_pid = PIDController(kp=20.0, ki=0.1, kd=0.2, dt=time_step)
    w_pid = PIDController(kp=50.0, ki=0.1, kd=0.2, dt=time_step)

    urdf_path = "sr_description/urdf/ball.urdf"
    mesh_dir = "sr_description"

    sim = RobotDynamics(urdf_path=urdf_path, shell_radius=0.4, dt=time_step)
    visualizer = RobotVisualizer(urdf_path=urdf_path, mesh_dir=mesh_dir)

    screen = pygame.display.set_mode((1100, 650))
    pygame.display.set_caption("Spherical Robot Controller")

    data_logger = DataLogger(max_points=300)
    path_manager = PathManager()
    path_following = False
    status_message = ""
    status_timestamp = 0.0

    planar_estimator = PlanarStateEstimator()
    mpc_dt = 0.02
    planar_mpc = PlanarMPC(dt=mpc_dt, horizon=25)
    mpc_update_stride = max(1, int(round(planar_mpc.dt / time_step)))
    mpc_solution = None

    clock = pygame.time.Clock()
    running = True
    last_time = time.time()
    loop_counter = 0

    print("Simulation started...")
    print("Plot legend: green=desired, red=actual, blue=output")

    while running:
        events = pygame.event.get()
        cursor_world: Optional[np.ndarray] = None

        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    path_manager.clear()
                    path_following = False
                    status_message = "Path cleared"
                    status_timestamp = time.time()
                elif event.key == pygame.K_RETURN:
                    path_manager.build_path()
                    status_message = "Path rebuilt"
                    status_timestamp = time.time()
                elif event.key == pygame.K_BACKSPACE:
                    path_manager.pop_last()
                    path_manager.build_path()
                elif event.key == pygame.K_SPACE:
                    if path_manager.has_path():
                        path_following = not path_following
                        status_message = "Follow mode" if path_following else "Edit mode"
                    else:
                        status_message = "No valid path to follow"
                    status_timestamp = time.time()
            elif event.type == pygame.MOUSEBUTTONDOWN and MAP_RECT.collidepoint(event.pos):
                world_point = map_pixel_to_world(event.pos)
                if event.button == 1:
                    path_manager.add_waypoint(world_point)
                    path_manager.build_path()
                elif event.button == 3:
                    path_manager.pop_last()
                    path_manager.build_path()
            elif event.type == pygame.MOUSEMOTION and MAP_RECT.collidepoint(event.pos):
                cursor_world = map_pixel_to_world(event.pos)
                if pygame.mouse.get_pressed()[0]:
                    path_manager.add_waypoint(cursor_world)
                    path_manager.build_path()

        if cursor_world is None and MAP_RECT.collidepoint(pygame.mouse.get_pos()):
            cursor_world = map_pixel_to_world(pygame.mouse.get_pos())

        screen.fill((20, 20, 30))

        planar_state = planar_estimator.extract(sim.q, sim.v)
        references = build_planar_reference(
            planar_state,
            path_manager,
            horizon=planar_mpc.horizon,
            dt=planar_mpc.dt,
        )

        if path_manager.has_path():
            if loop_counter % mpc_update_stride == 0 or mpc_solution is None:
                try:
                    mpc_solution = planar_mpc.solve(
                        current_state=planar_state.as_vector(),
                        reference_states=references,
                    )
                except np.linalg.LinAlgError:
                    mpc_solution = None
        else:
            mpc_solution = None

        first_velocity_hope, roll_hope = get_remote_input(joystick)
        if path_following and mpc_solution is not None:
            first_velocity_hope, roll_hope = compute_autonomous_command(
                planar_state=planar_state,
                mpc_solution=mpc_solution,
                references=references,
                shell_radius=sim.shell_radius,
                mpc_dt=planar_mpc.dt,
            )

        tau_actuators, actual_roll, actual_angular_velocity, actual_velocity = controller(
            sim, first_velocity_hope, roll_hope
        )

        draw_curve_plot(
            screen=screen,
            data_logger=data_logger,
            desired_value=first_velocity_hope / 10.0,
            actual_value=actual_velocity / 10.0,
            output_value=tau_actuators[0] / 50.0,
            position=(50, 50),
            size=(700, 350),
            title="Roll Angle Control (rad)",
            y_range=(-1.0, 1.0),
        )

        predicted_states = mpc_solution.predicted_states if mpc_solution is not None else None
        draw_path_panel(
            screen=screen,
            path_manager=path_manager,
            robot_xy=planar_state.position,
            path_following=path_following,
            cursor_world=cursor_world,
            predicted_states=predicted_states,
        )

        if status_message and time.time() - status_timestamp < 3.0:
            status_font = pygame.font.SysFont("Arial", 16)
            status_surface = status_font.render(status_message, True, (240, 240, 240))
            screen.blit(status_surface, (MAP_RECT.left, MAP_RECT.bottom + 120))

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
