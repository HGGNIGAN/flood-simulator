import os
import subprocess
from src.lisflood_docker.config_gen import generate_lisflood_settings


def run_lisflood_docker(
        host_paths, container_paths, lisflood_image="jrce1/lisflood:latest"
):
        """
        EXECUTION FUNCTION (REPLACEMENT) for "Step 2: Hydraulic Model"

        Args:
            host_paths: Dict containing paths on host machine
            container_paths: Dict containing paths inside container
            lisflood_image: Docker image name for LISFLOOD

        Returns:
            True if successful, False if failed
        """
        print("    Executing hydraulics (Step 2) via Docker...")

        # 1. Create settings XML file (on HOST machine)
        host_settings_file = host_paths["settings"]

        # Create settings directory if it doesn't exist
        os.makedirs(os.path.dirname(host_settings_file), exist_ok=True)

        generate_lisflood_settings(host_settings_file, container_paths, host_paths)

        # 2. Build Docker command
        host_data_dir = host_paths["base_data_dir"]
        container_data_dir = container_paths["base_data_dir"]
        container_settings_file = container_paths["settings"]

        # LISFLOOD Docker container expects the settings file path directly
        # Not a separate "lisflood" command
        docker_command = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{host_data_dir}:{container_data_dir}",
                lisflood_image,
                container_settings_file,  # Pass settings file directly
        ]

        print(f"    Docker command: {' '.join(docker_command)}")

        # 3. Run command
        try:
                subprocess.run(
                        docker_command, check=True, capture_output=True, text=True
                )
                print("    LISFLOOD ran successfully.")
                # Can add: print(result.stdout) to see full log

                # Verify output file was created
                expected_output = host_paths["output_raw"]
                if not os.path.exists(expected_output):
                        print(
                                f"    WARNING: Output file {expected_output} was not created."
                        )
                        return False

                print(f"    Output file created: {os.path.basename(expected_output)}")
                return True

        except subprocess.CalledProcessError as e:
                print(f"    ERROR RUNNING DOCKER (Exit Code: {e.returncode})")
                print(f"    STDERR:\n{e.stderr}")
                return False
        except FileNotFoundError:
                print("    ERROR: 'docker' not installed.")
                return False
        except Exception as e:
                print(f"    ERROR: Unexpected error: {e}")
                return False
