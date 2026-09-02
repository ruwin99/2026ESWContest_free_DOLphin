/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef enum
{
  MISSION_IDLE,
  INITIAL_SCAN,
  WAIT_CAPTURE_ACK,
  RETURNING,
  REALTIME_CLEANING,
  ACTUATOR_TEST,
  RESCAN_RETURN,
  RESCAN,
  MISSION_DONE,
  MISSION_FAULT
} MissionState;

typedef enum
{
  ACTUATOR_COMMAND_NONE,
  ACTUATOR_COMMAND_ON,
  ACTUATOR_COMMAND_OFF,
  ACTUATOR_COMMAND_CLEANER_PWM_33_3,
  ACTUATOR_COMMAND_CLEANER_PWM_55_6
} ActuatorCommand;

typedef enum
{
  UART_COMMAND_NONE,
  UART_COMMAND_START,
  UART_COMMAND_CAPTURE_OK,
  UART_COMMAND_ACTUATOR_TEST_START,
  UART_COMMAND_ACTUATOR_TEST_STOP,
  UART_COMMAND_FRONT_CLEANER_PWM_33_3,
  UART_COMMAND_FRONT_CLEANER_PWM_55_6,
  UART_COMMAND_FRONT_CLEANER_OFF,
  UART_COMMAND_FRONT_PUMP_ON,
  UART_COMMAND_FRONT_PUMP_OFF,
  UART_COMMAND_SIDE_CLEANER_PWM_33_3,
  UART_COMMAND_SIDE_CLEANER_PWM_55_6,
  UART_COMMAND_SIDE_CLEANER_OFF,
  UART_COMMAND_SIDE_PUMP_ON,
  UART_COMMAND_SIDE_PUMP_OFF,
  UART_COMMAND_COUNT
} UartCommand;

typedef enum
{
  UART_MESSAGE_STARTED,
  UART_MESSAGE_ACTUATOR_TEST_READY,
  UART_MESSAGE_CAMERA_CAPTURE,
  UART_MESSAGE_RETURN_START,
  UART_MESSAGE_REALTIME_START,
  UART_MESSAGE_RESCAN_RETURN_START,
  UART_MESSAGE_RESCAN_START,
  UART_MESSAGE_RESCAN_DONE,
  UART_MESSAGE_DONE,
  UART_MESSAGE_COUNT
} UartMessage;

typedef struct
{
  const char *text;
  uint16_t length;
} UartText;

typedef struct
{
  volatile int event;
  volatile int limit;
  int motor_direction;
} EncoderContext;

typedef struct
{
  volatile MissionState state;
  MissionState capture_resume_state;
} MissionContext;

typedef struct
{
  volatile uint32_t pending_commands;
  uint8_t match_index[UART_COMMAND_COUNT];
  volatile int rearm_required;
  uint8_t rx_byte;
} UartContext;

typedef struct
{
  volatile ActuatorCommand pending_command;
  volatile int active;
  volatile int hw_ready;
  volatile int last_on_tick;
} ActuatorContext;

typedef struct
{
  EncoderContext encoder;
  MissionContext mission;
  UartContext uart;
  ActuatorContext front_cleaner;
  ActuatorContext front_pump;
  ActuatorContext side_cleaner;
  ActuatorContext side_pump;
} RobotContext;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define PWM_DUTY 1600 /* Motor 2/base: 44.4%; motor 1: 47.1% after the 106% trim. */
#define LIVE_PWM_DUTY 800 /* Realtime motor 2/base: 22.2%; motor 1: 23.6% after the 106% trim. */
#define DC_MOTOR_PWM_DUTY_33_3 1200 /* raw TIM1 duty 33.3% */
#define DC_MOTOR_PWM_DUTY_55_6 2000 /* raw TIM1 duty 55.6% */
#define WATER_PUMP_PWM_DUTY 3000 /* raw TIM1 duty 83.3% (3600-count period). */
#define CLEANER_COMMAND_TIMEOUT_MS 1000
#define PUMP_COMMAND_TIMEOUT_MS 1000
#define ENCODER_SECTION_LIMIT 4
#define MOTOR_DIRECTION_FORWARD 1
#define MOTOR_DIRECTION_REVERSE -1
#define MOTOR_DIRECTION_STOPPED 0
#define ENCODER_1_DRIVE_PWM_PERCENT 106

/* Actuator hardware mapping follows the physical FRONT/SIDE wiring. */
#define FRONT_CLEANER_TIM_CHANNEL TIM_CHANNEL_4
#define FRONT_CLEANER_GPIO_PORT GPIOB
#define FRONT_CLEANER_IN1_PIN GPIO_PIN_13
#define FRONT_CLEANER_IN2_PIN GPIO_PIN_14

#define FRONT_WATER_PUMP_TIM_CHANNEL TIM_CHANNEL_1
#define FRONT_WATER_PUMP_GPIO_PORT GPIOA
#define FRONT_WATER_PUMP_IN1_PIN GPIO_PIN_5
#define FRONT_WATER_PUMP_IN2_PIN GPIO_PIN_6

#define SIDE_CLEANER_TIM_CHANNEL TIM_CHANNEL_2
#define SIDE_CLEANER_GPIO_PORT GPIOC
#define SIDE_CLEANER_IN1_PIN GPIO_PIN_13
#define SIDE_CLEANER_IN2_PIN GPIO_PIN_14

#define SIDE_WATER_PUMP_TIM_CHANNEL TIM_CHANNEL_3
#define SIDE_WATER_PUMP_GPIO_PORT GPIOC
#define SIDE_WATER_PUMP_IN1_PIN GPIO_PIN_8
#define SIDE_WATER_PUMP_IN2_PIN GPIO_PIN_9

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;

UART_HandleTypeDef huart2;
UART_HandleTypeDef huart3;

/* USER CODE BEGIN PV */
static RobotContext robot;
static const UartText uart_command_table[UART_COMMAND_COUNT] =
{
  [UART_COMMAND_NONE] = {NULL, 0U},
  [UART_COMMAND_START] = {"START\r\n", (uint16_t)(sizeof("START\r\n") - 1U)},
  [UART_COMMAND_CAPTURE_OK] = {"CAPTURE_OK\r\n", (uint16_t)(sizeof("CAPTURE_OK\r\n") - 1U)},
  [UART_COMMAND_ACTUATOR_TEST_START] = {"ACTUATOR_TEST_START\r\n", (uint16_t)(sizeof("ACTUATOR_TEST_START\r\n") - 1U)},
  [UART_COMMAND_ACTUATOR_TEST_STOP] = {"ACTUATOR_TEST_STOP\r\n", (uint16_t)(sizeof("ACTUATOR_TEST_STOP\r\n") - 1U)},
  [UART_COMMAND_FRONT_CLEANER_PWM_33_3] = {"FRONT_CLEANER_PWM_33_3\r\n", (uint16_t)(sizeof("FRONT_CLEANER_PWM_33_3\r\n") - 1U)},
  [UART_COMMAND_FRONT_CLEANER_PWM_55_6] = {"FRONT_CLEANER_PWM_55_6\r\n", (uint16_t)(sizeof("FRONT_CLEANER_PWM_55_6\r\n") - 1U)},
  [UART_COMMAND_FRONT_CLEANER_OFF] = {"FRONT_CLEANER_OFF\r\n", (uint16_t)(sizeof("FRONT_CLEANER_OFF\r\n") - 1U)},
  [UART_COMMAND_FRONT_PUMP_ON] = {"FRONT_PUMP_ON\r\n", (uint16_t)(sizeof("FRONT_PUMP_ON\r\n") - 1U)},
  [UART_COMMAND_FRONT_PUMP_OFF] = {"FRONT_PUMP_OFF\r\n", (uint16_t)(sizeof("FRONT_PUMP_OFF\r\n") - 1U)},
  [UART_COMMAND_SIDE_CLEANER_PWM_33_3] = {"SIDE_CLEANER_PWM_33_3\r\n", (uint16_t)(sizeof("SIDE_CLEANER_PWM_33_3\r\n") - 1U)},
  [UART_COMMAND_SIDE_CLEANER_PWM_55_6] = {"SIDE_CLEANER_PWM_55_6\r\n", (uint16_t)(sizeof("SIDE_CLEANER_PWM_55_6\r\n") - 1U)},
  [UART_COMMAND_SIDE_CLEANER_OFF] = {"SIDE_CLEANER_OFF\r\n", (uint16_t)(sizeof("SIDE_CLEANER_OFF\r\n") - 1U)},
  [UART_COMMAND_SIDE_PUMP_ON] = {"SIDE_PUMP_ON\r\n", (uint16_t)(sizeof("SIDE_PUMP_ON\r\n") - 1U)},
  [UART_COMMAND_SIDE_PUMP_OFF] = {"SIDE_PUMP_OFF\r\n", (uint16_t)(sizeof("SIDE_PUMP_OFF\r\n") - 1U)}
};

static const UartText uart_message_table[UART_MESSAGE_COUNT] =
{
  [UART_MESSAGE_STARTED] = {"STARTED\r\n", (uint16_t)(sizeof("STARTED\r\n") - 1U)},
  [UART_MESSAGE_ACTUATOR_TEST_READY] = {"ACTUATOR_TEST_READY\r\n", (uint16_t)(sizeof("ACTUATOR_TEST_READY\r\n") - 1U)},
  [UART_MESSAGE_CAMERA_CAPTURE] = {"CAMERA_CAPTURE\r\n", (uint16_t)(sizeof("CAMERA_CAPTURE\r\n") - 1U)},
  [UART_MESSAGE_RETURN_START] = {"RETURN_START\r\n", (uint16_t)(sizeof("RETURN_START\r\n") - 1U)},
  [UART_MESSAGE_REALTIME_START] = {"REALTIME_START\r\n", (uint16_t)(sizeof("REALTIME_START\r\n") - 1U)},
  [UART_MESSAGE_RESCAN_RETURN_START] = {"RESCAN_RETURN_START\r\n", (uint16_t)(sizeof("RESCAN_RETURN_START\r\n") - 1U)},
  [UART_MESSAGE_RESCAN_START] = {"RESCAN_START\r\n", (uint16_t)(sizeof("RESCAN_START\r\n") - 1U)},
  [UART_MESSAGE_RESCAN_DONE] = {"RESCAN_DONE\r\n", (uint16_t)(sizeof("RESCAN_DONE\r\n") - 1U)},
  [UART_MESSAGE_DONE] = {"DONE\r\n", (uint16_t)(sizeof("DONE\r\n") - 1U)}
};

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM3_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_TIM2_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_TIM4_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static void Cleaner_StopHardware(ActuatorContext *actuator,
                                 uint32_t tim_channel,
                                 GPIO_TypeDef *gpio_port,
                                 uint16_t in1_pin,
                                 uint16_t in2_pin)
{
  if (actuator->hw_ready != 0)
  {
    __HAL_TIM_SET_COMPARE(&htim1, tim_channel, 0);
    HAL_GPIO_WritePin(gpio_port, in1_pin | in2_pin, GPIO_PIN_RESET);
  }
  actuator->active = 0;
}

static void Cleaner_RunHardware(ActuatorContext *actuator,
                                uint32_t tim_channel,
                                GPIO_TypeDef *gpio_port,
                                uint16_t in1_pin,
                                uint16_t in2_pin,
                                uint16_t pwm_duty)
{
  if (actuator->hw_ready == 0)
  {
    return;
  }

  if (actuator->active == 0)
  {
    /* Select direction only while this role's PWM output is at zero. */
    __HAL_TIM_SET_COMPARE(&htim1, tim_channel, 0);
    HAL_GPIO_WritePin(gpio_port, in1_pin | in2_pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(gpio_port, in1_pin, GPIO_PIN_SET);
  }
  __HAL_TIM_SET_COMPARE(&htim1, tim_channel, pwm_duty);
  actuator->active = 1;
}

static void Pump_StopHardware(ActuatorContext *actuator,
                              uint32_t tim_channel,
                              GPIO_TypeDef *gpio_port,
                              uint16_t in1_pin,
                              uint16_t in2_pin)
{
  if (actuator->hw_ready != 0)
  {
    __HAL_TIM_SET_COMPARE(&htim1, tim_channel, 0);
    HAL_GPIO_WritePin(gpio_port, in1_pin | in2_pin, GPIO_PIN_RESET);
  }
  actuator->active = 0;
}

static void Pump_StartHardware(ActuatorContext *actuator,
                               uint32_t tim_channel,
                               GPIO_TypeDef *gpio_port,
                               uint16_t in1_pin,
                               uint16_t in2_pin)
{
  if (actuator->hw_ready == 0)
  {
    return;
  }

  if (actuator->active == 0)
  {
    /* Select direction only while this role's PWM output is at zero. */
    __HAL_TIM_SET_COMPARE(&htim1, tim_channel, 0);
    HAL_GPIO_WritePin(gpio_port, in1_pin | in2_pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(gpio_port, in1_pin, GPIO_PIN_SET);
  }
  __HAL_TIM_SET_COMPARE(&htim1, tim_channel, WATER_PUMP_PWM_DUTY);
  actuator->active = 1;
}

static void Front_Cleaner_Stop(void)
{
  Cleaner_StopHardware(&robot.front_cleaner, FRONT_CLEANER_TIM_CHANNEL,
                       FRONT_CLEANER_GPIO_PORT, FRONT_CLEANER_IN1_PIN,
                       FRONT_CLEANER_IN2_PIN);
}

static void Front_Cleaner_Run(uint16_t pwm_duty)
{
  Cleaner_RunHardware(&robot.front_cleaner, FRONT_CLEANER_TIM_CHANNEL,
                      FRONT_CLEANER_GPIO_PORT, FRONT_CLEANER_IN1_PIN,
                      FRONT_CLEANER_IN2_PIN, pwm_duty);
}

static void Front_Pump_Stop(void)
{
  Pump_StopHardware(&robot.front_pump, FRONT_WATER_PUMP_TIM_CHANNEL,
                    FRONT_WATER_PUMP_GPIO_PORT, FRONT_WATER_PUMP_IN1_PIN,
                    FRONT_WATER_PUMP_IN2_PIN);
}

static void Front_Pump_Start(void)
{
  Pump_StartHardware(&robot.front_pump, FRONT_WATER_PUMP_TIM_CHANNEL,
                     FRONT_WATER_PUMP_GPIO_PORT, FRONT_WATER_PUMP_IN1_PIN,
                     FRONT_WATER_PUMP_IN2_PIN);
}

static void Side_Cleaner_Stop(void)
{
  Cleaner_StopHardware(&robot.side_cleaner, SIDE_CLEANER_TIM_CHANNEL,
                       SIDE_CLEANER_GPIO_PORT, SIDE_CLEANER_IN1_PIN,
                       SIDE_CLEANER_IN2_PIN);
}

static void Side_Cleaner_Run(uint16_t pwm_duty)
{
  Cleaner_RunHardware(&robot.side_cleaner, SIDE_CLEANER_TIM_CHANNEL,
                      SIDE_CLEANER_GPIO_PORT, SIDE_CLEANER_IN1_PIN,
                      SIDE_CLEANER_IN2_PIN, pwm_duty);
}

static void Side_Pump_Stop(void)
{
  Pump_StopHardware(&robot.side_pump, SIDE_WATER_PUMP_TIM_CHANNEL,
                    SIDE_WATER_PUMP_GPIO_PORT, SIDE_WATER_PUMP_IN1_PIN,
                    SIDE_WATER_PUMP_IN2_PIN);
}

static void Side_Pump_Start(void)
{
  Pump_StartHardware(&robot.side_pump, SIDE_WATER_PUMP_TIM_CHANNEL,
                     SIDE_WATER_PUMP_GPIO_PORT, SIDE_WATER_PUMP_IN1_PIN,
                     SIDE_WATER_PUMP_IN2_PIN);
}

static void CleaningActuators_StopAll(void)
{
  Front_Cleaner_Stop();
  Front_Pump_Stop();
  Side_Cleaner_Stop();
  Side_Pump_Stop();
}

static int Actuator_ControlIsAllowed(void)
{
  return (((robot.mission.state == REALTIME_CLEANING) ||
           (robot.mission.state == ACTUATOR_TEST)) &&
          (robot.encoder.event == 0));
}

static void Actuator_QueueCommand(ActuatorContext *actuator,
                                  ActuatorCommand command)
{
  uint32_t primask = __get_PRIMASK();

  /* Keep the OFF latch atomic against UART ISR updates. */
  __disable_irq();
  /* OFF stays latched until main consumes it; later ON cannot overwrite it. */
  if ((command == ACTUATOR_COMMAND_OFF) ||
      (actuator->pending_command != ACTUATOR_COMMAND_OFF))
  {
    actuator->pending_command = command;
  }
  if (primask == 0U)
  {
    __enable_irq();
  }
}

static ActuatorCommand Actuator_TakePendingCommand(
    ActuatorContext *actuator)
{
  uint32_t primask;
  ActuatorCommand command;

  if (actuator->pending_command == ACTUATOR_COMMAND_NONE)
  {
    return ACTUATOR_COMMAND_NONE;
  }

  primask = __get_PRIMASK();
  /* Read-and-clear atomically against UART ISR command updates. */
  __disable_irq();
  command = actuator->pending_command;
  actuator->pending_command = ACTUATOR_COMMAND_NONE;
  if (primask == 0U)
  {
    __enable_irq();
  }

  return command;
}

static void Motor_Stop(void)
{
  if (htim2.Instance == TIM2)
  {
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, 0);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, 0);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, 0);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, 0);
  }
}

static void Motor_Run(int direction, int pwm_duty)
{
  int encoder_1_pwm_duty =
      (pwm_duty * ENCODER_1_DRIVE_PWM_PERCENT + 50) / 100;

  Motor_Stop();

  if (direction > 0)
  {
    /* Encoder drive 1 is wired with opposite polarity and has a 6% PWM trim. */
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, encoder_1_pwm_duty);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, pwm_duty);
  }
  else if (direction < 0)
  {
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, encoder_1_pwm_duty);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, pwm_duty);
  }
}

static void Motor_StartNextEncoderSection(int direction, int pwm_duty)
{
  if ((direction == MOTOR_DIRECTION_STOPPED) ||
      (robot.mission.state == WAIT_CAPTURE_ACK))
  {
    Motor_Stop();
    return;
  }

  /* Forward must count up on TIM3; reverse then counts down. */
  __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);

  if (direction > 0)
  {
    __HAL_TIM_SET_COUNTER(&htim3, 0);
  }
  else
  {
    __HAL_TIM_SET_COUNTER(&htim3, __HAL_TIM_GET_AUTORELOAD(&htim3));
  }

  __HAL_TIM_CLEAR_FLAG(&htim3, TIM_FLAG_UPDATE);
  HAL_NVIC_ClearPendingIRQ(TIM3_IRQn);

  Motor_Run(direction, pwm_duty);

  __HAL_TIM_ENABLE_IT(&htim3, TIM_IT_UPDATE);
}

static void UART_QueueCommand(UartCommand command)
{
  uint32_t command_bit;

  if ((command <= UART_COMMAND_NONE) || (command >= UART_COMMAND_COUNT))
  {
    return;
  }

  if ((((command == UART_COMMAND_FRONT_CLEANER_PWM_33_3) ||
        (command == UART_COMMAND_FRONT_CLEANER_PWM_55_6)) &&
       ((robot.uart.pending_commands &
         (1UL << UART_COMMAND_FRONT_CLEANER_OFF)) != 0U)) ||
      ((command == UART_COMMAND_FRONT_PUMP_ON) &&
       ((robot.uart.pending_commands &
         (1UL << UART_COMMAND_FRONT_PUMP_OFF)) != 0U)) ||
      (((command == UART_COMMAND_SIDE_CLEANER_PWM_33_3) ||
        (command == UART_COMMAND_SIDE_CLEANER_PWM_55_6)) &&
       ((robot.uart.pending_commands &
         (1UL << UART_COMMAND_SIDE_CLEANER_OFF)) != 0U)) ||
      ((command == UART_COMMAND_SIDE_PUMP_ON) &&
       ((robot.uart.pending_commands &
         (1UL << UART_COMMAND_SIDE_PUMP_OFF)) != 0U)))
  {
    return;
  }

  switch (command)
  {
    case UART_COMMAND_ACTUATOR_TEST_START:
    case UART_COMMAND_ACTUATOR_TEST_STOP:
      /* Test mode transitions supersede queued motion/actuator commands. */
      robot.uart.pending_commands = 0U;
      break;

    case UART_COMMAND_FRONT_PUMP_ON:
    case UART_COMMAND_SIDE_PUMP_ON:
    case UART_COMMAND_START:
    case UART_COMMAND_CAPTURE_OK:
      break;

    case UART_COMMAND_FRONT_CLEANER_OFF:
      robot.uart.pending_commands &=
          ~(1UL << UART_COMMAND_FRONT_CLEANER_PWM_33_3);
      robot.uart.pending_commands &=
          ~(1UL << UART_COMMAND_FRONT_CLEANER_PWM_55_6);
      break;

    case UART_COMMAND_FRONT_CLEANER_PWM_33_3:
    case UART_COMMAND_FRONT_CLEANER_PWM_55_6:
      robot.uart.pending_commands &=
          ~(1UL << UART_COMMAND_FRONT_CLEANER_PWM_33_3);
      robot.uart.pending_commands &=
          ~(1UL << UART_COMMAND_FRONT_CLEANER_PWM_55_6);
      break;

    case UART_COMMAND_FRONT_PUMP_OFF:
      robot.uart.pending_commands &= ~(1UL << UART_COMMAND_FRONT_PUMP_ON);
      break;

    case UART_COMMAND_SIDE_CLEANER_OFF:
      robot.uart.pending_commands &=
          ~(1UL << UART_COMMAND_SIDE_CLEANER_PWM_33_3);
      robot.uart.pending_commands &=
          ~(1UL << UART_COMMAND_SIDE_CLEANER_PWM_55_6);
      break;

    case UART_COMMAND_SIDE_CLEANER_PWM_33_3:
    case UART_COMMAND_SIDE_CLEANER_PWM_55_6:
      robot.uart.pending_commands &=
          ~(1UL << UART_COMMAND_SIDE_CLEANER_PWM_33_3);
      robot.uart.pending_commands &=
          ~(1UL << UART_COMMAND_SIDE_CLEANER_PWM_55_6);
      break;

    case UART_COMMAND_SIDE_PUMP_OFF:
      robot.uart.pending_commands &= ~(1UL << UART_COMMAND_SIDE_PUMP_ON);
      break;

    case UART_COMMAND_NONE:
    case UART_COMMAND_COUNT:
    default:
      return;
  }

  command_bit = 1UL << (uint32_t)command;
  robot.uart.pending_commands |= command_bit;
}

static UartCommand UART_TakePendingCommand(void)
{
  uint32_t primask;
  uint32_t command_bit = 0U;
  UartCommand command = UART_COMMAND_NONE;

  primask = __get_PRIMASK();
  __disable_irq();

  if ((robot.uart.pending_commands &
       (1UL << UART_COMMAND_ACTUATOR_TEST_STOP)) != 0U)
  {
    command = UART_COMMAND_ACTUATOR_TEST_STOP;
  }
  else if ((robot.uart.pending_commands &
       (1UL << UART_COMMAND_FRONT_CLEANER_OFF)) != 0U)
  {
    command = UART_COMMAND_FRONT_CLEANER_OFF;
  }
  else if ((robot.uart.pending_commands &
            (1UL << UART_COMMAND_FRONT_PUMP_OFF)) != 0U)
  {
    command = UART_COMMAND_FRONT_PUMP_OFF;
  }
  else if ((robot.uart.pending_commands &
            (1UL << UART_COMMAND_SIDE_CLEANER_OFF)) != 0U)
  {
    command = UART_COMMAND_SIDE_CLEANER_OFF;
  }
  else if ((robot.uart.pending_commands &
            (1UL << UART_COMMAND_SIDE_PUMP_OFF)) != 0U)
  {
    command = UART_COMMAND_SIDE_PUMP_OFF;
  }
  else if ((robot.uart.pending_commands &
            (1UL << UART_COMMAND_ACTUATOR_TEST_START)) != 0U)
  {
    command = UART_COMMAND_ACTUATOR_TEST_START;
  }
  else if ((robot.uart.pending_commands &
            (1UL << UART_COMMAND_CAPTURE_OK)) != 0U)
  {
    command = UART_COMMAND_CAPTURE_OK;
  }
  else if ((robot.uart.pending_commands &
            (1UL << UART_COMMAND_START)) != 0U)
  {
    command = UART_COMMAND_START;
  }
  else if ((robot.uart.pending_commands &
            (1UL << UART_COMMAND_FRONT_CLEANER_PWM_55_6)) != 0U)
  {
    command = UART_COMMAND_FRONT_CLEANER_PWM_55_6;
  }
  else if ((robot.uart.pending_commands &
            (1UL << UART_COMMAND_FRONT_CLEANER_PWM_33_3)) != 0U)
  {
    command = UART_COMMAND_FRONT_CLEANER_PWM_33_3;
  }
  else if ((robot.uart.pending_commands &
            (1UL << UART_COMMAND_FRONT_PUMP_ON)) != 0U)
  {
    command = UART_COMMAND_FRONT_PUMP_ON;
  }
  else if ((robot.uart.pending_commands &
            (1UL << UART_COMMAND_SIDE_CLEANER_PWM_55_6)) != 0U)
  {
    command = UART_COMMAND_SIDE_CLEANER_PWM_55_6;
  }
  else if ((robot.uart.pending_commands &
            (1UL << UART_COMMAND_SIDE_CLEANER_PWM_33_3)) != 0U)
  {
    command = UART_COMMAND_SIDE_CLEANER_PWM_33_3;
  }
  else if ((robot.uart.pending_commands &
            (1UL << UART_COMMAND_SIDE_PUMP_ON)) != 0U)
  {
    command = UART_COMMAND_SIDE_PUMP_ON;
  }

  if (command != UART_COMMAND_NONE)
  {
    command_bit = 1UL << (uint32_t)command;
    robot.uart.pending_commands &= ~command_bit;
  }

  if (primask == 0U)
  {
    __enable_irq();
  }

  return command;
}

static void UART_QueueFailSafeAllOff(void)
{
  uint32_t primask = __get_PRIMASK();

  __disable_irq();
  robot.uart.pending_commands = 0U;
  Actuator_QueueCommand(&robot.front_cleaner, ACTUATOR_COMMAND_OFF);
  Actuator_QueueCommand(&robot.front_pump, ACTUATOR_COMMAND_OFF);
  Actuator_QueueCommand(&robot.side_cleaner, ACTUATOR_COMMAND_OFF);
  Actuator_QueueCommand(&robot.side_pump, ACTUATOR_COMMAND_OFF);
  if (primask == 0U)
  {
    __enable_irq();
  }
}

static void UART_ParseByte(uint8_t byte)
{
  UartCommand command;

  for (command = UART_COMMAND_START;
       command < UART_COMMAND_COUNT;
       command = (UartCommand)((int)command + 1))
  {
    const UartText *definition = &uart_command_table[command];
    uint8_t match_index = robot.uart.match_index[command];

    if (((command == UART_COMMAND_START) &&
         (robot.mission.state != MISSION_IDLE)) ||
        ((command == UART_COMMAND_CAPTURE_OK) &&
         (robot.mission.state != WAIT_CAPTURE_ACK)) ||
        ((command == UART_COMMAND_ACTUATOR_TEST_START) &&
         (robot.mission.state != MISSION_IDLE)) ||
        ((command == UART_COMMAND_ACTUATOR_TEST_STOP) &&
         (robot.mission.state != ACTUATOR_TEST)))
    {
      robot.uart.match_index[command] = 0U;
      continue;
    }

    if (byte == (uint8_t)definition->text[match_index])
    {
      match_index++;
      if (match_index >= definition->length)
      {
        match_index = 0U;
        UART_QueueCommand(command);
      }
    }
    else if (byte == (uint8_t)definition->text[0])
    {
      match_index = 1U;
    }
    else
    {
      match_index = 0U;
    }

    robot.uart.match_index[command] = match_index;
  }
}

static void UART_ResetMatchers(void)
{
  UartCommand command;

  for (command = UART_COMMAND_START;
       command < UART_COMMAND_COUNT;
       command = (UartCommand)((int)command + 1))
  {
    robot.uart.match_index[command] = 0U;
  }
}

static void UART_SendMessage(UartMessage message)
{
  const UartText *entry;

  if (message >= UART_MESSAGE_COUNT)
  {
    Error_Handler();
  }

  entry = &uart_message_table[message];
  if (HAL_UART_Transmit(&huart2, (const uint8_t *)entry->text,
                        entry->length, 100) != HAL_OK)
  {
    Error_Handler();
  }
}

static void Mission_SetState(MissionState next_state)
{
  MissionState previous_state = robot.mission.state;

  if (previous_state == next_state)
  {
    return;
  }

  if (next_state == WAIT_CAPTURE_ACK)
  {
    if ((previous_state != INITIAL_SCAN) && (previous_state != RESCAN))
    {
      next_state = MISSION_FAULT;
    }
    else
    {
      robot.mission.capture_resume_state = previous_state;
    }
  }

  robot.mission.state = next_state;
  /* Every mission transition starts from a known four-channel OFF state. */
  CleaningActuators_StopAll();

  switch (robot.mission.state)
  {
    case MISSION_IDLE:
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      robot.encoder.motor_direction = MOTOR_DIRECTION_STOPPED;
      robot.encoder.event = 0;
      robot.encoder.limit = 0;
      robot.mission.capture_resume_state = INITIAL_SCAN;
      break;

    case INITIAL_SCAN:
      if (previous_state == MISSION_IDLE)
      {
        robot.encoder.event = 0;
        robot.encoder.limit = 0;
        UART_SendMessage(UART_MESSAGE_STARTED);
      }
      robot.encoder.motor_direction = MOTOR_DIRECTION_FORWARD;
      Motor_StartNextEncoderSection(robot.encoder.motor_direction, PWM_DUTY);
      break;

    case WAIT_CAPTURE_ACK:
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      UART_SendMessage(UART_MESSAGE_CAMERA_CAPTURE);
      break;

    case RETURNING:
      robot.encoder.limit = 0;
      robot.encoder.motor_direction = MOTOR_DIRECTION_REVERSE;
      UART_SendMessage(UART_MESSAGE_RETURN_START);
      Motor_StartNextEncoderSection(robot.encoder.motor_direction, PWM_DUTY);
      break;

    case REALTIME_CLEANING:
      robot.encoder.limit = 0;
      robot.encoder.motor_direction = MOTOR_DIRECTION_FORWARD;
      UART_SendMessage(UART_MESSAGE_REALTIME_START);
      Motor_StartNextEncoderSection(robot.encoder.motor_direction,
                                    LIVE_PWM_DUTY);
      break;

    case ACTUATOR_TEST:
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      robot.encoder.motor_direction = MOTOR_DIRECTION_STOPPED;
      robot.encoder.event = 0;
      robot.encoder.limit = 0;
      UART_SendMessage(UART_MESSAGE_ACTUATOR_TEST_READY);
      break;

    case RESCAN_RETURN:
      robot.encoder.limit = 0;
      robot.encoder.motor_direction = MOTOR_DIRECTION_REVERSE;
      UART_SendMessage(UART_MESSAGE_RESCAN_RETURN_START);
      Motor_StartNextEncoderSection(robot.encoder.motor_direction, PWM_DUTY);
      break;

    case RESCAN:
      if (previous_state == RESCAN_RETURN)
      {
        robot.encoder.limit = 0;
        robot.encoder.motor_direction = MOTOR_DIRECTION_FORWARD;
        UART_SendMessage(UART_MESSAGE_RESCAN_START);
      }
      Motor_StartNextEncoderSection(robot.encoder.motor_direction, PWM_DUTY);
      break;

    case MISSION_DONE:
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      robot.encoder.motor_direction = MOTOR_DIRECTION_STOPPED;
      UART_SendMessage(UART_MESSAGE_RESCAN_DONE);
      UART_SendMessage(UART_MESSAGE_DONE);
      break;

    case MISSION_FAULT:
    default:
      robot.mission.state = MISSION_FAULT;
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      robot.encoder.motor_direction = MOTOR_DIRECTION_STOPPED;
      break;
  }
}

static void Mission_HandleUartCommand(UartCommand command)
{
  switch (command)
  {
    case UART_COMMAND_START:
      if (robot.mission.state == MISSION_IDLE)
      {
        Mission_SetState(INITIAL_SCAN);
      }
      break;

    case UART_COMMAND_CAPTURE_OK:
      if (robot.mission.state != WAIT_CAPTURE_ACK)
      {
        break;
      }

      if (robot.mission.capture_resume_state == INITIAL_SCAN)
      {
        if (robot.encoder.limit >= ENCODER_SECTION_LIMIT)
        {
          Mission_SetState(RETURNING);
        }
        else
        {
          Mission_SetState(INITIAL_SCAN);
        }
      }
      else if (robot.mission.capture_resume_state == RESCAN)
      {
        if (robot.encoder.limit >= ENCODER_SECTION_LIMIT)
        {
          Mission_SetState(MISSION_DONE);
        }
        else
        {
          Mission_SetState(RESCAN);
        }
      }
      else
      {
        Mission_SetState(MISSION_FAULT);
      }
      break;

    case UART_COMMAND_ACTUATOR_TEST_START:
      if (robot.mission.state == MISSION_IDLE)
      {
        /* Drop any commands sent before the READY handshake. */
        UART_QueueFailSafeAllOff();
        Mission_SetState(ACTUATOR_TEST);
      }
      break;

    case UART_COMMAND_ACTUATOR_TEST_STOP:
      if (robot.mission.state == ACTUATOR_TEST)
      {
        UART_QueueFailSafeAllOff();
        Mission_SetState(MISSION_IDLE);
      }
      break;

    case UART_COMMAND_FRONT_CLEANER_PWM_33_3:
      Actuator_QueueCommand(&robot.front_cleaner,
                            ACTUATOR_COMMAND_CLEANER_PWM_33_3);
      break;

    case UART_COMMAND_FRONT_CLEANER_PWM_55_6:
      Actuator_QueueCommand(&robot.front_cleaner,
                            ACTUATOR_COMMAND_CLEANER_PWM_55_6);
      break;

    case UART_COMMAND_FRONT_CLEANER_OFF:
      Actuator_QueueCommand(&robot.front_cleaner, ACTUATOR_COMMAND_OFF);
      break;

    case UART_COMMAND_FRONT_PUMP_ON:
      Actuator_QueueCommand(&robot.front_pump, ACTUATOR_COMMAND_ON);
      break;

    case UART_COMMAND_FRONT_PUMP_OFF:
      Actuator_QueueCommand(&robot.front_pump, ACTUATOR_COMMAND_OFF);
      break;

    case UART_COMMAND_SIDE_CLEANER_PWM_33_3:
      Actuator_QueueCommand(&robot.side_cleaner,
                            ACTUATOR_COMMAND_CLEANER_PWM_33_3);
      break;

    case UART_COMMAND_SIDE_CLEANER_PWM_55_6:
      Actuator_QueueCommand(&robot.side_cleaner,
                            ACTUATOR_COMMAND_CLEANER_PWM_55_6);
      break;

    case UART_COMMAND_SIDE_CLEANER_OFF:
      Actuator_QueueCommand(&robot.side_cleaner, ACTUATOR_COMMAND_OFF);
      break;

    case UART_COMMAND_SIDE_PUMP_ON:
      Actuator_QueueCommand(&robot.side_pump, ACTUATOR_COMMAND_ON);
      break;

    case UART_COMMAND_SIDE_PUMP_OFF:
      Actuator_QueueCommand(&robot.side_pump, ACTUATOR_COMMAND_OFF);
      break;

    case UART_COMMAND_NONE:
    case UART_COMMAND_COUNT:
    default:
      break;
  }
}

static void Mission_Update(void)
{
  UartCommand command;

  while ((command = UART_TakePendingCommand()) != UART_COMMAND_NONE)
  {
    Mission_HandleUartCommand(command);
  }

  switch (robot.mission.state)
  {
    case MISSION_IDLE:
      break;

    case INITIAL_SCAN:
      if (robot.encoder.event != 0)
      {
        robot.encoder.event = 0;
        robot.encoder.limit++;
        Mission_SetState(WAIT_CAPTURE_ACK);
      }
      break;

    case WAIT_CAPTURE_ACK:
      break;

    case RETURNING:
      if (robot.encoder.event != 0)
      {
        robot.encoder.event = 0;
        __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
        Motor_Stop();
        robot.encoder.limit++;

        if (robot.encoder.limit >= ENCODER_SECTION_LIMIT)
        {
          Mission_SetState(REALTIME_CLEANING);
        }
        else
        {
          Motor_StartNextEncoderSection(robot.encoder.motor_direction,
                                        PWM_DUTY);
        }
      }
      break;

    case REALTIME_CLEANING:
      if (robot.encoder.event != 0)
      {
        robot.encoder.event = 0;

        if (robot.encoder.limit >= ENCODER_SECTION_LIMIT)
        {
          __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
          Motor_Stop();
          Mission_SetState(RESCAN_RETURN);
        }
      }
      break;

    case ACTUATOR_TEST:
      /* This service state must never energize the drive motor. */
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      robot.encoder.motor_direction = MOTOR_DIRECTION_STOPPED;
      robot.encoder.event = 0;
      break;

    case RESCAN_RETURN:
      if (robot.encoder.event != 0)
      {
        robot.encoder.event = 0;

        if (robot.encoder.limit >= ENCODER_SECTION_LIMIT)
        {
          __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
          Motor_Stop();
          Mission_SetState(RESCAN);
        }
      }
      break;

    case RESCAN:
      if (robot.encoder.event != 0)
      {
        robot.encoder.event = 0;
        robot.encoder.limit++;
        Mission_SetState(WAIT_CAPTURE_ACK);
      }
      break;

    case MISSION_DONE:
      Mission_SetState(MISSION_IDLE);
      break;

    case MISSION_FAULT:
      robot.uart.pending_commands = 0U;
      robot.encoder.event = 0;
      break;

    default:
      Mission_SetState(MISSION_FAULT);
      break;
  }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  ActuatorCommand front_cleaner_command;
  ActuatorCommand front_pump_command;
  ActuatorCommand side_cleaner_command;
  ActuatorCommand side_pump_command;

  robot.encoder.motor_direction = MOTOR_DIRECTION_STOPPED;
  robot.mission.state = MISSION_IDLE;
  robot.mission.capture_resume_state = INITIAL_SCAN;
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM1_Init();
  MX_TIM3_Init();
  MX_USART3_UART_Init();
  MX_TIM2_Init();
  MX_USART2_UART_Init();
  MX_TIM4_Init();
  /* USER CODE BEGIN 2 */
  /* TIM1 drives two cleaner motors and two water pumps at 1 kHz. */
  __HAL_TIM_SET_COMPARE(&htim1, FRONT_CLEANER_TIM_CHANNEL, 0);
  __HAL_TIM_SET_COMPARE(&htim1, FRONT_WATER_PUMP_TIM_CHANNEL, 0);
  __HAL_TIM_SET_COMPARE(&htim1, SIDE_CLEANER_TIM_CHANNEL, 0);
  __HAL_TIM_SET_COMPARE(&htim1, SIDE_WATER_PUMP_TIM_CHANNEL, 0);

  if ((HAL_TIM_PWM_Start(&htim1, FRONT_CLEANER_TIM_CHANNEL) != HAL_OK) ||
      (HAL_TIM_PWM_Start(&htim1, FRONT_WATER_PUMP_TIM_CHANNEL) != HAL_OK) ||
      (HAL_TIM_PWM_Start(&htim1, SIDE_CLEANER_TIM_CHANNEL) != HAL_OK) ||
      (HAL_TIM_PWM_Start(&htim1, SIDE_WATER_PUMP_TIM_CHANNEL) != HAL_OK))
  {
    Error_Handler();
  }
  /* TIM2 drives both MDD3A channels at 20 kHz: M1A/M1B, M2A/M2B. */
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, 0);
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, 0);
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, 0);
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, 0);
  if ((HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1) != HAL_OK) ||
      (HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2) != HAL_OK) ||
      (HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3) != HAL_OK) ||
      (HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_4) != HAL_OK))
  {
    Error_Handler();
  }
  robot.front_cleaner.hw_ready = 1;
  robot.front_pump.hw_ready = 1;
  robot.side_cleaner.hw_ready = 1;
  robot.side_pump.hw_ready = 1;
  CleaningActuators_StopAll();
  if (HAL_TIM_Encoder_Start(&htim3, TIM_CHANNEL_ALL) != HAL_OK)
  {
    Error_Handler();
  }
  /* TIM4 encoder counting is started but not read; capture/section IRQs use TIM3. */
  if (HAL_TIM_Encoder_Start(&htim4, TIM_CHANNEL_ALL) != HAL_OK)
  {
    Error_Handler();
  }

  Motor_Stop();
  HAL_NVIC_ClearPendingIRQ(USART2_IRQn);
  if (HAL_UART_Receive_IT(&huart2, &robot.uart.rx_byte, 1) != HAL_OK)
  {
    Error_Handler();
  }

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    if ((robot.uart.rearm_required != 0) &&
        (huart2.RxState == HAL_UART_STATE_READY))
    {
      robot.uart.rearm_required = 0;
      if (HAL_UART_Receive_IT(&huart2, &robot.uart.rx_byte, 1) != HAL_OK)
      {
        robot.uart.rearm_required = 1;
        UART_ResetMatchers();
        UART_QueueFailSafeAllOff();
        CleaningActuators_StopAll();
      }
    }

    front_cleaner_command = ACTUATOR_COMMAND_NONE;
    if (robot.front_cleaner.pending_command != ACTUATOR_COMMAND_NONE)
    {
      front_cleaner_command =
          Actuator_TakePendingCommand(&robot.front_cleaner);
    }
    if (front_cleaner_command == ACTUATOR_COMMAND_OFF)
    {
      Front_Cleaner_Stop();
    }
    else if ((front_cleaner_command ==
              ACTUATOR_COMMAND_CLEANER_PWM_33_3) ||
             (front_cleaner_command ==
              ACTUATOR_COMMAND_CLEANER_PWM_55_6))
    {
      if (Actuator_ControlIsAllowed() != 0)
      {
        robot.front_cleaner.last_on_tick = (int)HAL_GetTick();
        if (front_cleaner_command == ACTUATOR_COMMAND_CLEANER_PWM_55_6)
        {
          Front_Cleaner_Run(DC_MOTOR_PWM_DUTY_55_6);
        }
        else
        {
          Front_Cleaner_Run(DC_MOTOR_PWM_DUTY_33_3);
        }
      }
      else
      {
        Front_Cleaner_Stop();
      }
    }

    front_pump_command = ACTUATOR_COMMAND_NONE;
    if (robot.front_pump.pending_command != ACTUATOR_COMMAND_NONE)
    {
      front_pump_command = Actuator_TakePendingCommand(&robot.front_pump);
    }
    if (front_pump_command == ACTUATOR_COMMAND_OFF)
    {
      Front_Pump_Stop();
    }
    else if (front_pump_command == ACTUATOR_COMMAND_ON)
    {
      if (Actuator_ControlIsAllowed() != 0)
      {
        robot.front_pump.last_on_tick = (int)HAL_GetTick();
        if (robot.front_pump.active == 0)
        {
          Front_Pump_Start();
        }
      }
      else
      {
        Front_Pump_Stop();
      }
    }

    side_cleaner_command = ACTUATOR_COMMAND_NONE;
    if (robot.side_cleaner.pending_command != ACTUATOR_COMMAND_NONE)
    {
      side_cleaner_command =
          Actuator_TakePendingCommand(&robot.side_cleaner);
    }
    if (side_cleaner_command == ACTUATOR_COMMAND_OFF)
    {
      Side_Cleaner_Stop();
    }
    else if ((side_cleaner_command ==
              ACTUATOR_COMMAND_CLEANER_PWM_33_3) ||
             (side_cleaner_command ==
              ACTUATOR_COMMAND_CLEANER_PWM_55_6))
    {
      if (Actuator_ControlIsAllowed() != 0)
      {
        robot.side_cleaner.last_on_tick = (int)HAL_GetTick();
        if (side_cleaner_command == ACTUATOR_COMMAND_CLEANER_PWM_55_6)
        {
          Side_Cleaner_Run(DC_MOTOR_PWM_DUTY_55_6);
        }
        else
        {
          Side_Cleaner_Run(DC_MOTOR_PWM_DUTY_33_3);
        }
      }
      else
      {
        Side_Cleaner_Stop();
      }
    }

    side_pump_command = ACTUATOR_COMMAND_NONE;
    if (robot.side_pump.pending_command != ACTUATOR_COMMAND_NONE)
    {
      side_pump_command = Actuator_TakePendingCommand(&robot.side_pump);
    }
    if (side_pump_command == ACTUATOR_COMMAND_OFF)
    {
      Side_Pump_Stop();
    }
    else if (side_pump_command == ACTUATOR_COMMAND_ON)
    {
      if (Actuator_ControlIsAllowed() != 0)
      {
        robot.side_pump.last_on_tick = (int)HAL_GetTick();
        if (robot.side_pump.active == 0)
        {
          Side_Pump_Start();
        }
      }
      else
      {
        Side_Pump_Stop();
      }
    }

    if ((robot.front_cleaner.active != 0) &&
        ((Actuator_ControlIsAllowed() == 0) ||
         ((uint32_t)(HAL_GetTick() -
                     (uint32_t)robot.front_cleaner.last_on_tick) >=
          CLEANER_COMMAND_TIMEOUT_MS)))
    {
      Front_Cleaner_Stop();
    }

    if ((robot.front_pump.active != 0) &&
        ((Actuator_ControlIsAllowed() == 0) ||
         ((uint32_t)(HAL_GetTick() -
                     (uint32_t)robot.front_pump.last_on_tick) >=
          PUMP_COMMAND_TIMEOUT_MS)))
    {
      Front_Pump_Stop();
    }

    if ((robot.side_cleaner.active != 0) &&
        ((Actuator_ControlIsAllowed() == 0) ||
         ((uint32_t)(HAL_GetTick() -
                     (uint32_t)robot.side_cleaner.last_on_tick) >=
          CLEANER_COMMAND_TIMEOUT_MS)))
    {
      Side_Cleaner_Stop();
    }

    if ((robot.side_pump.active != 0) &&
        ((Actuator_ControlIsAllowed() == 0) ||
         ((uint32_t)(HAL_GetTick() -
                     (uint32_t)robot.side_pump.last_on_tick) >=
          PUMP_COMMAND_TIMEOUT_MS)))
    {
      Side_Pump_Stop();
    }

    Mission_Update();
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }

  /** Enables the Clock Security System
  */
  HAL_RCC_EnableCSS();
}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 19;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 3599;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = 0;
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
  if (HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */
  HAL_TIM_MspPostInit(&htim1);

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 3599;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit(&htim2);

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 0;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 600;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI1;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 4;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 4;
  if (HAL_TIM_Encoder_Init(&htim3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 0;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 600;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 4;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 4;
  if (HAL_TIM_Encoder_Init(&htim4, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_8|GPIO_PIN_9, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5|GPIO_PIN_6, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_13|GPIO_PIN_14, GPIO_PIN_RESET);

  /*Configure GPIO pins : PC13 PC14 PC8 PC9 */
  GPIO_InitStruct.Pin = GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_8|GPIO_PIN_9;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : PA5 PA6 */
  GPIO_InitStruct.Pin = GPIO_PIN_5|GPIO_PIN_6;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PB13 PB14 */
  GPIO_InitStruct.Pin = GPIO_PIN_13|GPIO_PIN_14;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    if (huart->ErrorCode == HAL_UART_ERROR_NONE)
    {
      UART_ParseByte(robot.uart.rx_byte);

      if (HAL_UART_Receive_IT(&huart2, &robot.uart.rx_byte, 1) != HAL_OK)
      {
        robot.uart.rearm_required = 1;
        UART_QueueFailSafeAllOff();
      }
      else
      {
        robot.uart.rearm_required = 0;
      }
    }
    else
    {
      UART_ResetMatchers();
      robot.uart.rearm_required = 1;
      UART_QueueFailSafeAllOff();
    }
  }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    UART_ResetMatchers();
    UART_QueueFailSafeAllOff();
    if (huart->RxState == HAL_UART_STATE_READY)
    {
      robot.uart.rearm_required = 1;
    }
  }
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  if (htim->Instance == TIM3)
  {
    if ((robot.mission.state == REALTIME_CLEANING) ||
        (robot.mission.state == RESCAN_RETURN))
    {
      /* Continuous passes keep PWM running through ARR events 1 to 3. */
      robot.encoder.limit++;
      if (robot.encoder.limit >= ENCODER_SECTION_LIMIT)
      {
        __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
        Motor_Stop();
        if (robot.mission.state == REALTIME_CLEANING)
        {
          CleaningActuators_StopAll();
        }
        robot.encoder.event = 1;
      }
    }
    else if ((robot.mission.state == INITIAL_SCAN) ||
             (robot.mission.state == RETURNING) ||
             (robot.mission.state == RESCAN))
    {
      /* Stop at each section; main handles the state transition and UART. */
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      robot.encoder.event = 1;
    }
    else
    {
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
    }
  }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  robot.mission.state = MISSION_FAULT;
  robot.encoder.motor_direction = MOTOR_DIRECTION_STOPPED;
  if (htim2.Instance == TIM2)
  {
    Motor_Stop();
  }
  CleaningActuators_StopAll();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
