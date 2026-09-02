from __future__ import annotations

import re
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


class StmFrontSideActuatorProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = STM_MAIN.read_text(encoding="utf-8")

    @classmethod
    def function_source(cls, signature: str) -> str:
        start = cls.source.index(signature)
        opening_brace = cls.source.index("{", start)
        depth = 0
        for index in range(opening_brace, len(cls.source)):
            character = cls.source[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return cls.source[start : index + 1]
        raise AssertionError(f"Unclosed function: {signature}")

    @staticmethod
    def case_source(function_source: str, case_name: str, next_case: str) -> str:
        start = function_source.index(f"case {case_name}:")
        end = function_source.index(f"case {next_case}:", start)
        return function_source[start:end]

    def test_protocol_has_exact_role_specific_commands_and_no_legacy_aliases(
        self,
    ) -> None:
        table_start = self.source.index(
            "static const UartText uart_command_table[UART_COMMAND_COUNT]"
        )
        table_end = self.source.index(
            "static const UartText uart_message_table[UART_MESSAGE_COUNT]",
            table_start,
        )
        table = self.source[table_start:table_end]
        commands = set(re.findall(r'\{"([A-Z0-9_]+)\\r\\n"', table))
        actuator_commands = commands - {
            "START",
            "CAPTURE_OK",
            "ACTUATOR_TEST_START",
            "ACTUATOR_TEST_STOP",
        }
        self.assertEqual(
            actuator_commands,
            {
                "FRONT_CLEANER_PWM_33_3",
                "FRONT_CLEANER_PWM_55_6",
                "FRONT_CLEANER_OFF",
                "FRONT_PUMP_ON",
                "FRONT_PUMP_OFF",
                "SIDE_CLEANER_PWM_33_3",
                "SIDE_CLEANER_PWM_55_6",
                "SIDE_CLEANER_OFF",
                "SIDE_PUMP_ON",
                "SIDE_PUMP_OFF",
            },
        )
        for legacy_command in (
            "CLEANER_ON",
            "CLEANER_PWM_12_5",
            "CLEANER_PWM_25",
            "FRONT_CLEANER_PWM_12_5",
            "FRONT_CLEANER_PWM_25",
            "SIDE_CLEANER_PWM_12_5",
            "SIDE_CLEANER_PWM_25",
            "CLEANER_OFF",
            "PUMP_ON",
            "PUMP_OFF",
        ):
            self.assertNotIn(f'{{"{legacy_command}\\r\\n"', table)

    def test_actuator_test_requires_explicit_handshake_and_stops_drive(self) -> None:
        self.assertIn('"ACTUATOR_TEST_START\\r\\n"', self.source)
        self.assertIn('"ACTUATOR_TEST_STOP\\r\\n"', self.source)
        self.assertIn('"ACTUATOR_TEST_READY\\r\\n"', self.source)

        set_state = self.function_source("static void Mission_SetState")
        test_state = self.case_source(
            set_state,
            "ACTUATOR_TEST",
            "RESCAN_RETURN",
        )
        self.assertIn("__HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);", test_state)
        self.assertIn("Motor_Stop();", test_state)
        self.assertIn(
            "robot.encoder.motor_direction = MOTOR_DIRECTION_STOPPED;",
            test_state,
        )
        self.assertIn(
            "UART_SendMessage(UART_MESSAGE_ACTUATOR_TEST_READY);",
            test_state,
        )

        update = self.function_source("static void Mission_Update")
        update_test_state = self.case_source(
            update,
            "ACTUATOR_TEST",
            "RESCAN_RETURN",
        )
        self.assertIn("__HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);", update_test_state)
        self.assertIn("Motor_Stop();", update_test_state)

    def test_actuator_test_start_stop_are_fail_safe_state_transitions(self) -> None:
        handler = self.function_source("static void Mission_HandleUartCommand")
        start = self.case_source(
            handler,
            "UART_COMMAND_ACTUATOR_TEST_START",
            "UART_COMMAND_ACTUATOR_TEST_STOP",
        )
        self.assertIn("robot.mission.state == MISSION_IDLE", start)
        self.assertIn("UART_QueueFailSafeAllOff();", start)
        self.assertIn("Mission_SetState(ACTUATOR_TEST);", start)

        stop = self.case_source(
            handler,
            "UART_COMMAND_ACTUATOR_TEST_STOP",
            "UART_COMMAND_FRONT_CLEANER_PWM_33_3",
        )
        self.assertIn("robot.mission.state == ACTUATOR_TEST", stop)
        self.assertIn("UART_QueueFailSafeAllOff();", stop)
        self.assertIn("Mission_SetState(MISSION_IDLE);", stop)

        take = self.function_source("static UartCommand UART_TakePendingCommand")
        self.assertLess(
            take.index("UART_COMMAND_ACTUATOR_TEST_STOP"),
            take.index("UART_COMMAND_FRONT_CLEANER_OFF"),
        )

    def test_actuator_outputs_are_allowed_only_in_realtime_or_test_state(
        self,
    ) -> None:
        allowed = self.function_source("static int Actuator_ControlIsAllowed")
        self.assertIn("robot.mission.state == REALTIME_CLEANING", allowed)
        self.assertIn("robot.mission.state == ACTUATOR_TEST", allowed)
        self.assertIn("robot.encoder.event == 0", allowed)

        main_start = self.source.index("int main(void)")
        main_loop = self.source[
            main_start : self.source.index("void SystemClock_Config(void)", main_start)
        ]
        self.assertEqual(main_loop.count("Actuator_ControlIsAllowed() != 0"), 4)
        self.assertEqual(main_loop.count("Actuator_ControlIsAllowed() == 0"), 4)

    def test_uart_receive_failures_request_four_channel_fail_safe(self) -> None:
        rx_complete = self.function_source("void HAL_UART_RxCpltCallback")
        uart_error = self.function_source("void HAL_UART_ErrorCallback")
        self.assertIn("UART_QueueFailSafeAllOff();", rx_complete)
        self.assertIn("UART_QueueFailSafeAllOff();", uart_error)

        fail_safe = self.function_source("static void UART_QueueFailSafeAllOff")
        self.assertEqual(fail_safe.count("ACTUATOR_COMMAND_OFF"), 4)

    def test_front_and_side_cleaner_hardware_mapping(self) -> None:
        expected_defines = {
            "FRONT_CLEANER_TIM_CHANNEL": "TIM_CHANNEL_4",
            "FRONT_CLEANER_GPIO_PORT": "GPIOB",
            "FRONT_CLEANER_IN1_PIN": "GPIO_PIN_13",
            "FRONT_CLEANER_IN2_PIN": "GPIO_PIN_14",
            "FRONT_WATER_PUMP_TIM_CHANNEL": "TIM_CHANNEL_1",
            "FRONT_WATER_PUMP_GPIO_PORT": "GPIOA",
            "FRONT_WATER_PUMP_IN1_PIN": "GPIO_PIN_5",
            "FRONT_WATER_PUMP_IN2_PIN": "GPIO_PIN_6",
            "SIDE_CLEANER_TIM_CHANNEL": "TIM_CHANNEL_2",
            "SIDE_CLEANER_GPIO_PORT": "GPIOC",
            "SIDE_CLEANER_IN1_PIN": "GPIO_PIN_13",
            "SIDE_CLEANER_IN2_PIN": "GPIO_PIN_14",
            "SIDE_WATER_PUMP_TIM_CHANNEL": "TIM_CHANNEL_3",
            "SIDE_WATER_PUMP_GPIO_PORT": "GPIOC",
            "SIDE_WATER_PUMP_IN1_PIN": "GPIO_PIN_8",
            "SIDE_WATER_PUMP_IN2_PIN": "GPIO_PIN_9",
        }
        for name, value in expected_defines.items():
            self.assertRegex(
                self.source,
                rf"#define\s+{re.escape(name)}\s+{re.escape(value)}\b",
            )

        role_wrappers = {
            "static void Front_Cleaner_Run": (
                "&robot.front_cleaner",
                "FRONT_CLEANER_TIM_CHANNEL",
            ),
            "static void Front_Pump_Start": (
                "&robot.front_pump",
                "FRONT_WATER_PUMP_TIM_CHANNEL",
            ),
            "static void Side_Cleaner_Run": (
                "&robot.side_cleaner",
                "SIDE_CLEANER_TIM_CHANNEL",
            ),
            "static void Side_Pump_Start": (
                "&robot.side_pump",
                "SIDE_WATER_PUMP_TIM_CHANNEL",
            ),
        }
        for signature, expected_text in role_wrappers.items():
            wrapper = self.function_source(signature)
            for text in expected_text:
                self.assertIn(text, wrapper)

    def test_cleaner_compare_values_match_current_front_and_side_calibration(
        self,
    ) -> None:
        self.assertRegex(
            self.source,
            r"#define\s+DC_MOTOR_PWM_DUTY_33_3\s+1200\b",
        )
        self.assertRegex(
            self.source,
            r"#define\s+DC_MOTOR_PWM_DUTY_55_6\s+2000\b",
        )
        cleaner_run = self.function_source("static void Cleaner_RunHardware")
        self.assertEqual(
            cleaner_run.count(
                "__HAL_TIM_SET_COMPARE(&htim1, tim_channel, pwm_duty);"
            ),
            1,
        )
        self.assertIn("Front_Cleaner_Run(DC_MOTOR_PWM_DUTY_33_3)", self.source)
        self.assertIn("Front_Cleaner_Run(DC_MOTOR_PWM_DUTY_55_6)", self.source)
        self.assertIn("Side_Cleaner_Run(DC_MOTOR_PWM_DUTY_33_3)", self.source)
        self.assertIn("Side_Cleaner_Run(DC_MOTOR_PWM_DUTY_55_6)", self.source)

    def test_four_contexts_have_independent_watchdogs_and_global_fail_safe(
        self,
    ) -> None:
        self.assertIn("#define CLEANER_COMMAND_TIMEOUT_MS 1000", self.source)
        self.assertIn("#define PUMP_COMMAND_TIMEOUT_MS 1000", self.source)
        contexts = {
            "front_cleaner": "CLEANER_COMMAND_TIMEOUT_MS",
            "front_pump": "PUMP_COMMAND_TIMEOUT_MS",
            "side_cleaner": "CLEANER_COMMAND_TIMEOUT_MS",
            "side_pump": "PUMP_COMMAND_TIMEOUT_MS",
        }
        for context, timeout in contexts.items():
            self.assertIn(f"ActuatorContext {context};", self.source)
            self.assertIn(f"robot.{context}.last_on_tick =", self.source)
            timeout_check = re.compile(
                rf"robot\.{context}\.last_on_tick\).*?{timeout}",
                flags=re.DOTALL,
            )
            self.assertRegex(self.source, timeout_check)

        fail_safe = self.function_source("static void UART_QueueFailSafeAllOff")
        self.assertIn("robot.uart.pending_commands = 0U;", fail_safe)
        for context in contexts:
            self.assertIn(
                f"Actuator_QueueCommand(&robot.{context}, ACTUATOR_COMMAND_OFF)",
                fail_safe,
            )

        stop_all = self.function_source("static void CleaningActuators_StopAll")
        for stop_function in (
            "Front_Cleaner_Stop();",
            "Front_Pump_Stop();",
            "Side_Cleaner_Stop();",
            "Side_Pump_Stop();",
        ):
            self.assertIn(stop_function, stop_all)

    def test_uart_off_latches_are_role_isolated_and_have_priority(self) -> None:
        queue = self.function_source("static void UART_QueueCommand")
        take = self.function_source("static UartCommand UART_TakePendingCommand")

        isolated_off_cases = {
            "UART_COMMAND_FRONT_CLEANER_OFF": (
                "UART_COMMAND_FRONT_CLEANER_PWM_33_3",
                "UART_COMMAND_FRONT_CLEANER_PWM_55_6",
                "UART_COMMAND_FRONT_CLEANER_PWM_33_3",
            ),
            "UART_COMMAND_FRONT_PUMP_OFF": (
                "UART_COMMAND_FRONT_PUMP_ON",
                "UART_COMMAND_FRONT_PUMP_ON",
                "UART_COMMAND_SIDE_CLEANER_OFF",
            ),
            "UART_COMMAND_SIDE_CLEANER_OFF": (
                "UART_COMMAND_SIDE_CLEANER_PWM_33_3",
                "UART_COMMAND_SIDE_CLEANER_PWM_55_6",
                "UART_COMMAND_SIDE_CLEANER_PWM_33_3",
            ),
            "UART_COMMAND_SIDE_PUMP_OFF": (
                "UART_COMMAND_SIDE_PUMP_ON",
                "UART_COMMAND_SIDE_PUMP_ON",
                "UART_COMMAND_NONE",
            ),
        }
        for off_command, (first_active, second_active, next_case) in (
            isolated_off_cases.items()
        ):
            block = self.case_source(queue, off_command, next_case)
            self.assertIn(first_active, block)
            self.assertIn(second_active, block)
            other_role = "SIDE_" if "FRONT_" in off_command else "FRONT_"
            self.assertNotIn(other_role, block)

        active_commands = (
            "UART_COMMAND_FRONT_CLEANER_PWM_55_6",
            "UART_COMMAND_FRONT_CLEANER_PWM_33_3",
            "UART_COMMAND_FRONT_PUMP_ON",
            "UART_COMMAND_SIDE_CLEANER_PWM_55_6",
            "UART_COMMAND_SIDE_CLEANER_PWM_33_3",
            "UART_COMMAND_SIDE_PUMP_ON",
        )
        first_active_index = min(take.index(command) for command in active_commands)
        for off_command in isolated_off_cases:
            self.assertLess(take.index(off_command), first_active_index)

        self.assertIn(
            "1UL << UART_COMMAND_FRONT_CLEANER_OFF",
            queue,
        )
        self.assertIn("1UL << UART_COMMAND_FRONT_PUMP_OFF", queue)
        self.assertIn("1UL << UART_COMMAND_SIDE_CLEANER_OFF", queue)
        self.assertIn("1UL << UART_COMMAND_SIDE_PUMP_OFF", queue)

    def test_role_commands_route_to_only_the_matching_context(self) -> None:
        handler = self.function_source("static void Mission_HandleUartCommand")
        routes = {
            "UART_COMMAND_FRONT_CLEANER_PWM_33_3": "robot.front_cleaner",
            "UART_COMMAND_FRONT_CLEANER_PWM_55_6": "robot.front_cleaner",
            "UART_COMMAND_FRONT_CLEANER_OFF": "robot.front_cleaner",
            "UART_COMMAND_FRONT_PUMP_ON": "robot.front_pump",
            "UART_COMMAND_FRONT_PUMP_OFF": "robot.front_pump",
            "UART_COMMAND_SIDE_CLEANER_PWM_33_3": "robot.side_cleaner",
            "UART_COMMAND_SIDE_CLEANER_PWM_55_6": "robot.side_cleaner",
            "UART_COMMAND_SIDE_CLEANER_OFF": "robot.side_cleaner",
            "UART_COMMAND_SIDE_PUMP_ON": "robot.side_pump",
            "UART_COMMAND_SIDE_PUMP_OFF": "robot.side_pump",
        }
        command_names = tuple(routes)
        for index, command_name in enumerate(command_names):
            next_case = (
                command_names[index + 1]
                if index + 1 < len(command_names)
                else "UART_COMMAND_NONE"
            )
            block = self.case_source(handler, command_name, next_case)
            self.assertIn(f"&{routes[command_name]}", block)
            for other_context in set(routes.values()) - {routes[command_name]}:
                self.assertNotIn(f"&{other_context}", block)

    def test_actuator_off_latch_is_atomic_against_uart_interrupts(self) -> None:
        function_source = self.function_source("static void Actuator_QueueCommand")
        disable_index = function_source.index("__disable_irq();")
        assignment_index = function_source.index("actuator->pending_command = command;")
        enable_index = function_source.index("__enable_irq();")
        self.assertLess(disable_index, assignment_index)
        self.assertLess(assignment_index, enable_index)


if __name__ == "__main__":
    unittest.main()
