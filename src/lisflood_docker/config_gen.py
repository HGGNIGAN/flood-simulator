import os
import xml.etree.ElementTree as ET


def generate_lisflood_settings(settings_file_path, container_paths, host_paths):
        """
        Create a settings XML file according to standard LISFLOOD template.

        LISFLOOD expects a very specific XML structure with many required sections.
        This function creates a minimal but functional configuration.

        Args:
            settings_file_path: Path to XML file on host
            container_paths: Dict containing paths inside container
            host_paths: Dict containing paths on host (to verify files exist)
        """
        print(f"... Creating LISFLOOD settings file at: {settings_file_path}")

        # Verify input files exist on host
        required_files = {
                "DEM": host_paths["dem"],
                "Roughness": host_paths["roughness"],
                "Precipitation": host_paths["rain"],
        }

        for file_type, file_path in required_files.items():
                if not os.path.exists(file_path):
                        print(
                                f"    WARNING: {file_type} file does not exist: {file_path}"
                        )

        # These paths are PATHS INSIDE CONTAINER
        dem_path = container_paths["dem"]
        roughness_path = container_paths["roughness"]
        rain_path = container_paths["rain"]
        output_path = container_paths["output"]

        # Get base directories
        output_dir = os.path.dirname(output_path)
        static_dir = os.path.dirname(dem_path)

        # Create XML structure according to LISFLOOD standard
        root = ET.Element("lfsettings")

        # lfoptions section - REQUIRED by LISFLOOD
        lfoptions = ET.SubElement(root, "lfoptions")

        # Basic options - minimal set for flood simulation
        ET.SubElement(lfoptions, "setoption", name="InitLisflood", choice="0")
        ET.SubElement(lfoptions, "setoption", name="readNetcdfStack", choice="0")
        ET.SubElement(lfoptions, "setoption", name="gridSizeUserDefined", choice="1")

        # lfuser section - Path variables and calendar settings
        lfuser = ET.SubElement(root, "lfuser")

        # Create group for paths and time settings
        path_group = ET.SubElement(lfuser, "group")

        # Path variables with comments
        path_root = ET.SubElement(
                path_group, "textvar", name="PathRoot", value=output_dir
        )
        path_root_comment = ET.SubElement(path_root, "comment")
        path_root_comment.text = "Root path for outputs"

        path_out = ET.SubElement(
                path_group, "textvar", name="PathOut", value=output_dir
        )
        path_out_comment = ET.SubElement(path_out, "comment")
        path_out_comment.text = "Output path"

        path_maps = ET.SubElement(
                path_group, "textvar", name="PathMaps", value=static_dir
        )
        path_maps_comment = ET.SubElement(path_maps, "comment")
        path_maps_comment.text = "Static maps path"

        # Calendar settings - REQUIRED by LISFLOOD
        calendar_convention = ET.SubElement(
                path_group,
                "textvar",
                name="CalendarConvention",
                value="proleptic_gregorian",
        )
        calendar_convention_comment = ET.SubElement(calendar_convention, "comment")
        calendar_convention_comment.text = "Calendar type"

        calendar_start = ET.SubElement(
                path_group, "textvar", name="CalendarDayStart", value="01/01/2020 00:00"
        )
        calendar_start_comment = ET.SubElement(calendar_start, "comment")
        calendar_start_comment.text = "Start of calendar"

        # Time step settings
        step_start = ET.SubElement(
                path_group, "textvar", name="StepStart", value="01/01/2020 00:00"
        )
        step_start_comment = ET.SubElement(step_start, "comment")
        step_start_comment.text = "Simulation start time"

        step_end = ET.SubElement(
                path_group, "textvar", name="StepEnd", value="01/01/2020 01:00"
        )
        step_end_comment = ET.SubElement(step_end, "comment")
        step_end_comment.text = "Simulation end time (1 hour simulation)"

        dt_sec = ET.SubElement(path_group, "textvar", name="DtSec", value="3600")
        dt_sec_comment = ET.SubElement(dt_sec, "comment")
        dt_sec_comment.text = "Time step in seconds (1 hour)"

        # Initial timestep - REQUIRED by LISFLOOD
        timestep_init = ET.SubElement(
                path_group, "textvar", name="timestepInit", value="01/01/2020 00:00"
        )
        timestep_init_comment = ET.SubElement(timestep_init, "comment")
        timestep_init_comment.text = "Initial time step for initial conditions"

        # lfbinding section - File bindings
        # In LISFLOOD, bindings reference variables defined in lfuser using $(VariableName) syntax
        lfbinding = ET.SubElement(root, "lfbinding")

        # Create a group for bindings
        binding_group = ET.SubElement(lfbinding, "group")

        # Calendar bindings - reference the user variables
        calendar_binding = ET.SubElement(
                binding_group,
                "textvar",
                name="CalendarConvention",
                value="$(CalendarConvention)",
        )
        calendar_binding_comment = ET.SubElement(calendar_binding, "comment")
        calendar_binding_comment.text = "Calendar convention"

        calendar_start_binding = ET.SubElement(
                binding_group,
                "textvar",
                name="CalendarDayStart",
                value="$(CalendarDayStart)",
        )
        calendar_start_binding_comment = ET.SubElement(
                calendar_start_binding, "comment"
        )
        calendar_start_binding_comment.text = "Reference day and time"

        dt_binding = ET.SubElement(
                binding_group, "textvar", name="DtSec", value="$(DtSec)"
        )
        dt_binding_comment = ET.SubElement(dt_binding, "comment")
        dt_binding_comment.text = "timestep [seconds]"

        step_start_binding = ET.SubElement(
                binding_group, "textvar", name="StepStart", value="$(StepStart)"
        )
        step_start_binding_comment = ET.SubElement(step_start_binding, "comment")
        step_start_binding_comment.text = "Number of first time step in simulation"

        step_end_binding = ET.SubElement(
                binding_group, "textvar", name="StepEnd", value="$(StepEnd)"
        )
        step_end_binding_comment = ET.SubElement(step_end_binding, "comment")
        step_end_binding_comment.text = "Number of last time step in simulation"

        # Initial timestep binding
        timestep_init_binding = ET.SubElement(
                binding_group, "textvar", name="timestepInit", value="$(timestepInit)"
        )
        timestep_init_binding_comment = ET.SubElement(timestep_init_binding, "comment")
        timestep_init_binding_comment.text = "Initial time step for initial conditions"

        # Precipitation maps - LISFLOOD uses prefix system
        precip_elem = ET.SubElement(
                binding_group, "textvar", name="PrecipitationMaps", value=rain_path
        )
        precip_comment = ET.SubElement(precip_elem, "comment")
        precip_comment.text = "precipitation [mm/day]"

        # DEM as Grad (gradient map)
        grad_elem = ET.SubElement(binding_group, "textvar", name="Grad", value=dem_path)
        grad_comment = ET.SubElement(grad_elem, "comment")
        grad_comment.text = "slope gradient from DEM"

        # Manning's roughness - LISFLOOD expects MapN for overland flow
        mapn_elem = ET.SubElement(
                binding_group, "textvar", name="MapN", value=roughness_path
        )
        mapn_comment = ET.SubElement(mapn_elem, "comment")
        mapn_comment.text = "Manning's roughness coefficient"

        # Output - Water depth
        output_elem = ET.SubElement(
                binding_group, "textvar", name="DischargeMaps", value=output_path
        )
        output_comment = ET.SubElement(output_elem, "comment")
        output_comment.text = "Output discharge/flood depth maps"

        # Lưu file XML
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)  # Format for readability
        tree.write(settings_file_path, encoding="utf-8", xml_declaration=True)
        print("    Successfully created settings file")
