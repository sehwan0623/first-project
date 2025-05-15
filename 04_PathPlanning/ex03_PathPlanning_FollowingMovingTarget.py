import numpy as np
import matplotlib.pyplot as plt
from lane_1 import lane  # 차선 함수: x에 대해 L/R 차선 y값 반환
from ex01_PathPlanning_BothLane import Global2Local, Polyfit, Polyval, BothLane2Path, VehicleModel_Lat, PurePursuit
# 여러 유틸리티 함수 및 클래스 (로컬 좌표 변환, 다항식 피팅, 차량 모델 등)

# 📦 선행차량의 과거 위치 저장 클래스
class LeadingVehiclePos(object):
    def __init__(self, num_data_store=5):
        self.max_num_array = num_data_store
        self.PosArray_Global = []  # 글로벌 좌표계 위치 저장

    def update(self, pos_global):
        if pos_global:
            self.PosArray_Global.append(pos_global[0])
        if len(self.PosArray_Global) > self.max_num_array:
            self.PosArray_Global.pop(0)  # 오래된 위치 제거 (FIFO)

# 📐 헤딩 각도 추정 함수 (3차 다항식 미분)
def HeadingAngleEstimation(coeff_path, PosArray):
    if coeff_path is None or len(coeff_path) != 4:
        return 0.0
    a3, a2, a1, _ = coeff_path  # 3차 방정식 계수
    x = 2.0  # 예측 지점 (lookahead point)
    dy_dx = 3*a3*x**2 + 2*a2*x + a1  # 도함수 계산
    heading = np.arctan(dy_dx)  # 기울기 → 각도(rad)
    print(f"[HeadingAngle] @x=2.0m → {heading:.3f} rad")
    return heading

# 📌 추종 경로 계산 (선행차 위치 기반으로 로컬 경로 다항식 구성)
def TargetFollowingPath(PosArray_Global, ego_yaw, ego_X, ego_Y):
    if len(PosArray_Global) < 3:
        return [0.0, 0.0, 0.0, 0.0]  # 포인트 부족 시 예외 처리

    # 전역 → 로컬 좌표계 변환
    local_points = Global2Local(PosArray_Global, ego_yaw, ego_X, ego_Y)
    arr = np.array(local_points)
    x = arr[:, 0]
    y = arr[:, 1]

    coeff_path = np.polyfit(x, y, 3).tolist()  # 3차 다항식 피팅
    HeadingAngleEstimation(coeff_path, local_points)  # 헤딩 계산
    return coeff_path

# 🚗 메인 실행부
if __name__ == "__main__":
    step_time = 0.1  # 샘플링 주기
    simulation_time = 30.0
    Vx = 3.0  # 차량 속도 (3 m/s 고정)

    # 도로 차선 정의
    X_lane = np.arange(0.0, 100.0, 0.1)
    Y_lane_L, Y_lane_R = lane(X_lane)  # 차선 좌우 y좌표 계산

    # 차량 초기화 (선행차: 원점, 추종차: -10m 뒤)
    leading_vehicle = VehicleModel_Lat(step_time, Vx)
    ego_vehicle = VehicleModel_Lat(step_time, Vx, Pos=[-10.0, 0.0, 0.0])

    # Pure Pursuit 제어기 초기화
    controller_lead = PurePursuit()
    controller_ego = PurePursuit()

    # 선행차 위치 기록 객체
    leading_vehicle_pos = LeadingVehiclePos()

    # 시각화용 경로 기록
    time = []
    X_lead = []
    Y_lead = []
    X_ego = []
    Y_ego = []

    plt.figure(figsize=(13, 2))  # 도로 형태에 맞게 가로형 출력

    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        X_lead.append(leading_vehicle.X)
        Y_lead.append(leading_vehicle.Y)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)

        # 차선 기준 5m 앞까지 좌우 차선 생성
        X_ref = np.arange(leading_vehicle.X, leading_vehicle.X + 5.0, 1.0)
        Y_ref_L, Y_ref_R = lane(X_ref)
        global_points_L = np.transpose(np.array([X_ref, Y_ref_L])).tolist()
        global_points_R = np.transpose(np.array([X_ref, Y_ref_R])).tolist()

        # 로컬 좌표계 변환 (선행차 기준)
        local_points_L = Global2Local(global_points_L, leading_vehicle.Yaw, leading_vehicle.X, leading_vehicle.Y)
        local_points_R = Global2Local(global_points_R, leading_vehicle.Yaw, leading_vehicle.X, leading_vehicle.Y)

        # 좌우 차선 → 중앙경로 다항식 생성
        coeff_L = Polyfit(local_points_L, num_order=3)
        coeff_R = Polyfit(local_points_R, num_order=3)
        coeff_path_lead = BothLane2Path(coeff_L, coeff_R)

        # 🟥 추종차는 선행차 위치만 추적
        pos_lead_global = [[leading_vehicle.X, leading_vehicle.Y]]
        leading_vehicle_pos.update(pos_lead_global)

        # 선행차 위치 기반 경로 피팅 (추종차 기준 로컬)
        coeff_path_ego = TargetFollowingPath(
            leading_vehicle_pos.PosArray_Global,
            ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)

        # Pure Pursuit 컨트롤 입력
        controller_lead.ControllerInput(coeff_path_lead, Vx)
        controller_ego.ControllerInput(coeff_path_ego, Vx)

        # 차량 상태 업데이트
        leading_vehicle.update(controller_lead.u, Vx)
        ego_vehicle.update(controller_ego.u, Vx)

        # 차량 시각화
        plt.plot(ego_vehicle.X, ego_vehicle.Y, 'bo')  # 파랑: ego
        plt.plot(leading_vehicle.X, leading_vehicle.Y, 'ro')  # 빨강: lead
        plt.axis("equal")
        plt.pause(0.01)

    # 시뮬레이션 종료 후 전체 경로 표시
    plt.show()