from interbotix_xs_modules.xs_launch import (
    declare_interbotix_xsarm_robot_description_launch_arguments,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import (
  IfCondition,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile


def launch_setup(context, *args, **kwargs):

    # Widowx 250s Arm (Robot)
    robot_model_launch_arg = LaunchConfiguration('robot_model')

    robot_name_launch_arg = LaunchConfiguration('robot_name')

    robot_modes_launch_arg = LaunchConfiguration('robot_modes')

    robot_description_launch_arg = LaunchConfiguration(
        'robot_description'
    )
    robot_launch_include = GroupAction(
        condition=IfCondition(LaunchConfiguration('use_robot')),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('interbotix_xsarm_control'),
                        'launch',
                        'xsarm_control.launch.py'
                    ])
                ]),
                launch_arguments={
                    'robot_model': robot_model_launch_arg,
                    'robot_name': robot_name_launch_arg,
                    'mode_configs': robot_modes_launch_arg,
                    'motor_configs': PathJoinSubstitution([
                        FindPackageShare('interbotix_xsarm_control'),
                        'config',
                        f'{robot_model_launch_arg.perform(context)}.yaml',
                    ]),
                    'use_rviz': 'false',
                    'robot_description': robot_description_launch_arg,
                }.items(),
            )
        ]
    )

    # Realsense Cameras
    rs_actions = []
    # camera_names = [
    #     LaunchConfiguration('cam_overhead_name'),
    #     LaunchConfiguration('cam_field_name'),
    #     LaunchConfiguration('cam_wrist_name'),
    # ]
    camera_names = [
        LaunchConfiguration('cam_field_name'),
        LaunchConfiguration('cam_wrist_name')
    ]
    camera_yaml_mapping = {
        'cam_overhead': 'rs_cam_overhead.yaml',
        'cam_field': 'rs_cam_field.yaml',
        'cam_wrist': 'rs_cam_wrist.yaml',
    }
    for camera_name in camera_names:
        camera_name_str = camera_name.perform(context)
        assert camera_name_str in camera_yaml_mapping, f'No yaml file for camera {camera_name_str}'
        yaml_file = camera_yaml_mapping.get(camera_name_str)
        
        rs_actions.append(
            Node(
                package='realsense2_camera',
                namespace=camera_name,
                name='camera',
                executable='realsense2_camera_node',
                parameters=[
                    ParameterFile(
                        param_file=PathJoinSubstitution([
                            FindPackageShare('robot'),
                            'config',
                            yaml_file,
                        ]),
                        allow_substs=True,
                    ),
                    {'initial_reset': True},
                ],
                output='screen',
            ),
        )
    realsense_ros_launch_includes_group_action = GroupAction(
      condition=IfCondition(LaunchConfiguration('use_cameras')),
      actions=rs_actions,
    )

    # Geogmagic Omni Touch
    got = []
    geomagic_omni_touch_node = Node(
        package="omni_common",
        executable="omni_state",
        output="screen",
    )
    got.append(geomagic_omni_touch_node)
    geomagic_touch_node = GroupAction(
        condition=IfCondition(LaunchConfiguration('use_omni')),
        actions=got,
    )

    return [
        robot_launch_include,
        geomagic_touch_node,
        realsense_ros_launch_includes_group_action,
    ]


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_model',
            default_value='wx250s',
            description='model type of the follower arms.'
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_name',
            default_value='robot',
            description='robot arm name',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_modes',
            default_value=PathJoinSubstitution([
                FindPackageShare('robot'),
                'config',
                'robot_modes.yaml',
            ]),
            description="the file path to the 'mode config' YAML file for the robot arm.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_robot',
            default_value='true',
            choices=('true', 'false'),
            description='if `true`, launches the Widowx 250s Robot Arm.',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_cameras',
            default_value='true',
            choices=('true', 'false'),
            description='if `true`, launches the camera drivers.',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_omni',
            default_value='true',
            choices=('true', 'false'),
            description='if `true`, launches the Geomagic Omni Touch.',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'cam_overhead_name',
            default_value='cam_overhead',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'cam_field_name',
            default_value='cam_field',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'cam_wrist_name',
            default_value='cam_wrist',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_joystick_teleop',
            default_value='false',
            choices=('true', 'false'),
            description='if `true`, launches a joystick teleop node for the base',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'rviz',
            default_value='false',
            choices=('true', 'false'),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_rvizconfig',
            default_value=PathJoinSubstitution([
                FindPackageShare('robot'),
                'rviz',
                'robot.rviz',
            ]),
        )
    )
    declared_arguments.extend(
        declare_interbotix_xsarm_robot_description_launch_arguments(
            robot_description_launch_config_name='robot_description',
            robot_model_launch_config_name='robot_model',
            robot_name_launch_config_name='robot_name',
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
