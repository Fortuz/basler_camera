from setuptools import find_packages, setup

package_name = 'basler_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fortuz',
    maintainer_email='fortuz19@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'basler_camera = basler_camera.camera_node:main',
            'calibration_node = basler_camera.calibration_node:main',
            'chessboard_calibration_node = basler_camera.chessboard_calibration_node:main',
            'yolo_node = basler_camera.yolo_node:main'
        ],
    },
)
