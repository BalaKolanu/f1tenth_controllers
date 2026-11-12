from setuptools import setup

package_name = 'f1tenth_controllers'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/mpc_steer_penalty_schedule.csv']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='nvidia@nvidia-desktop',
    description='F1TENTH MPC controller node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mpc_controller = f1tenth_controllers.mpc_controller:main',
            'pure_pursuit_controller = f1tenth_controllers.pure_pursuit_controller:main',
        ],
    },
)
