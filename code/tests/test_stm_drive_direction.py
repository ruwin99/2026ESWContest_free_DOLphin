from __future__ import annotations

import unittest
from pathlib import Path


STM_MAIN = (
    Path(__file__).resolve().parents[2]
    / "stm_code"
    / "code"
    / "encoder"
    / "Core"
    / "Src"
    / "main.c"
)


class StmDriveDirectionTests(unittest.TestCase):
    def test_drive_one_is_reversed_and_has_six_percent_pwm_trim(self) -> None:
        source = STM_MAIN.read_text(encoding="utf-8")
        self.assertIn("#define ENCODER_1_DRIVE_PWM_PERCENT 106", source)
        function_start = source.index("static void Motor_Run")
        function_end = source.index("static void Motor_StartNextEncoderSection")
        motor_run = source[function_start:function_end]

        self.assertIn(
            "(pwm_duty * ENCODER_1_DRIVE_PWM_PERCENT + 50) / 100",
            motor_run,
        )

        forward_start = motor_run.index("if (direction > 0)")
        reverse_start = motor_run.index("else if (direction < 0)")
        forward = motor_run[forward_start:reverse_start]
        reverse = motor_run[reverse_start:]

        self.assertIn("TIM_CHANNEL_2, encoder_1_pwm_duty", forward)
        self.assertIn("TIM_CHANNEL_3, pwm_duty", forward)
        self.assertNotIn("TIM_CHANNEL_1, pwm_duty", forward)
        self.assertNotIn("TIM_CHANNEL_4, pwm_duty", forward)

        self.assertIn("TIM_CHANNEL_1, encoder_1_pwm_duty", reverse)
        self.assertIn("TIM_CHANNEL_4, pwm_duty", reverse)
        self.assertNotIn("TIM_CHANNEL_2, pwm_duty", reverse)
        self.assertNotIn("TIM_CHANNEL_3, pwm_duty", reverse)

    def test_mission_uses_four_rail_sections_for_scan_and_return_boundaries(
        self,
    ) -> None:
        source = STM_MAIN.read_text(encoding="utf-8")
        self.assertIn("#define ENCODER_SECTION_LIMIT 4", source)

        command_handler_start = source.index(
            "static void Mission_HandleUartCommand(UartCommand command)"
        )
        mission_update_start = source.index("static void Mission_Update(void)")
        command_handler = source[command_handler_start:mission_update_start]
        self.assertEqual(
            command_handler.count(
                "robot.encoder.limit >= ENCODER_SECTION_LIMIT"
            ),
            2,
        )
        self.assertIn("Mission_SetState(RETURNING);", command_handler)
        self.assertIn("Mission_SetState(MISSION_DONE);", command_handler)

        mission_update_end = source.index("/* USER CODE END 0 */")
        mission_update = source[mission_update_start:mission_update_end]
        self.assertEqual(
            mission_update.count(
                "robot.encoder.limit >= ENCODER_SECTION_LIMIT"
            ),
            3,
        )
        self.assertIn("Mission_SetState(REALTIME_CLEANING);", mission_update)
        self.assertIn("Mission_SetState(RESCAN_RETURN);", mission_update)
        self.assertIn("Mission_SetState(RESCAN);", mission_update)


if __name__ == "__main__":
    unittest.main()
