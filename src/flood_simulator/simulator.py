"""
Optimized Flood Simulator
Combines physics-based hydraulics (LISFLOOD) with efficient computation (demo).
Implements Manning's equation, infiltration, and iterative flow routing.
"""

import rasterio
import numpy as np
import warnings

# Try to import numba for JIT acceleration (optional but recommended)
try:
        from numba import jit, prange

        NUMBA_AVAILABLE = True
except ImportError:
        print("    [WARNING] Numba not available. Install with: pip install numba")
        print("    [WARNING] Simulation will run without JIT acceleration (slower)")
        NUMBA_AVAILABLE = False

        # Dummy decorators if numba not available
        def jit(*args, **kwargs):
                def decorator(func):
                        return func

                return decorator

        prange = range

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- CONFIGURATION ---
TIME_STEP_SECONDS = 10.0  # Computational time step (seconds)
SIMULATION_DURATION_SECONDS = 7200.0  # Total simulation time (2 hours default)
CELL_SIZE = 30.0  # Grid cell size (meters)
GRAVITY = 9.81  # Gravitational acceleration (m/s²)
MIN_DEPTH_THRESHOLD = 0.001  # Minimum water depth to consider (meters)
MAX_ITERATIONS = int(SIMULATION_DURATION_SECONDS / TIME_STEP_SECONDS)


@jit(nopython=True, parallel=True, cache=True)
def calculate_manning_flow(h_current, dem, roughness, valid_mask, dt, dx, ny, nx):
        """
        Calculate water flow using Manning's equation with 4-neighbor D8 scheme.
        Optimized with Numba JIT compilation for parallel execution.

        Physics:
        Q = (1/n) * A * R_h^(2/3) * S^(1/2)
        where Q = flow, n = Manning's n, A = area, R_h = hydraulic radius, S = slope

        Args:
            h_current: Current water depth (m)
            dem: Digital elevation model (m)
            roughness: Manning's roughness coefficient
            valid_mask: Boolean mask for valid cells
            dt: Time step (seconds)
            dx: Cell size (meters)
            ny, nx: Grid dimensions

        Returns:
            Updated water depth array
        """
        h_new = h_current.copy()

        # Process each cell in parallel
        for i in prange(1, ny - 1):
                for j in prange(1, nx - 1):
                        if (
                                not valid_mask[i, j]
                                or h_current[i, j] < MIN_DEPTH_THRESHOLD
                        ):
                                continue

                        # Water surface elevation
                        wse_center = dem[i, j] + h_current[i, j]

                        # Check 4 neighbors (N, S, E, W)
                        neighbors = [
                                (i - 1, j),  # North
                                (i + 1, j),  # South
                                (i, j - 1),  # West
                                (i, j + 1),  # East
                        ]

                        total_outflow = 0.0

                        for ni, nj in neighbors:
                                if not valid_mask[ni, nj]:
                                        continue

                                # Water surface elevation of neighbor
                                wse_neighbor = dem[ni, nj] + h_current[ni, nj]

                                # Water surface slope (driving force)
                                slope = (wse_center - wse_neighbor) / dx

                                # Only flow downhill
                                if slope <= 0:
                                        continue

                                # Flow depth (depth at interface)
                                h_flow = max(h_current[i, j], h_current[ni, nj])

                                if h_flow < MIN_DEPTH_THRESHOLD:
                                        continue

                                # Average Manning's n between cells
                                n_avg = (roughness[i, j] + roughness[ni, nj]) / 2.0

                                # Manning's equation: v = (1/n) * R_h^(2/3) * S^(1/2)
                                # For wide shallow flow: R_h ≈ h_flow
                                velocity = (
                                        (1.0 / n_avg)
                                        * (h_flow ** (2.0 / 3.0))
                                        * (slope**0.5)
                                )

                                # Flow rate: Q = v * A, where A = h_flow * cell_width
                                flow_rate = velocity * h_flow * dx  # m³/s

                                # Volume transferred in this time step
                                volume = flow_rate * dt  # m³

                                # Convert to depth change
                                depth_change = volume / (dx * dx)  # m

                                # Limit outflow to available water
                                depth_change = min(depth_change, h_current[i, j] * 0.25)

                                total_outflow += depth_change

                        # Update water depth (cannot go negative)
                        h_new[i, j] = max(0.0, h_current[i, j] - total_outflow)

        return h_new


@jit(nopython=True, parallel=True, cache=True)
def calculate_infiltration(h_current, ksat, imperviousness, valid_mask, dt, ny, nx):
        """
        Calculate infiltration losses using Green-Ampt simplified model.
        Optimized with Numba JIT compilation.

        Args:
            h_current: Current water depth (m)
            ksat: Saturated hydraulic conductivity (m/s)
            imperviousness: Imperviousness fraction (0-1)
            valid_mask: Boolean mask
            dt: Time step (seconds)
            ny, nx: Grid dimensions

        Returns:
            Updated water depth after infiltration
        """
        h_new = h_current.copy()

        for i in prange(ny):
                for j in prange(nx):
                        if (
                                not valid_mask[i, j]
                                or h_current[i, j] < MIN_DEPTH_THRESHOLD
                        ):
                                continue

                        # Pervious fraction
                        pervious_frac = 1.0 - imperviousness[i, j]

                        # Maximum infiltration in this time step
                        max_infiltration = ksat[i, j] * dt * pervious_frac  # meters

                        # Actual infiltration (limited by available water)
                        infiltration = min(max_infiltration, h_current[i, j])

                        # Update depth
                        h_new[i, j] = max(0.0, h_current[i, j] - infiltration)

        return h_new


def demo_simulation(host_paths):
        """
        Physics-based flood simulation combining LISFLOOD methodology with optimized computation.

        Process:
        1. Load static data (DEM, roughness, hydraulic conductivity, imperviousness)
        2. Initialize water depth from precipitation
        3. Iterative time-stepping loop:
           - Calculate infiltration losses (Green-Ampt model)
           - Calculate surface flow using Manning's equation
           - Update water depths
        4. Save final flood depth map

        Args:
            host_paths: Dictionary containing file paths:
                - dem: Digital elevation model (m)
                - rain: Precipitation rate (mm/hr)
                - roughness: Manning's n coefficient
                - ksat (optional): Saturated hydraulic conductivity (mm/hr)
                - imperviousness (optional): Imperviousness fraction (0-1)
                - output_raw: Output file path

        Returns:
            bool: True if successful, False otherwise
        """
        try:
                print("    [SIMULATOR] Starting physics-based simulation...")
                if NUMBA_AVAILABLE:
                        print("    [SIMULATOR] Using Numba JIT acceleration (parallel)")

                # ==================== PHASE 1: LOAD STATIC DATA ====================

                # Load DEM
                with rasterio.open(host_paths["dem"]) as src:
                        profile = src.profile
                        dem = src.read(1).astype(np.float32)
                        nodata_val = src.nodata if src.nodata is not None else -9999.0
                        transform = src.transform

                ny, nx = dem.shape

                # Load precipitation (mm/hr)
                with rasterio.open(host_paths["rain"]) as src_rain:
                        precip_mmhr = src_rain.read(1).astype(np.float32)

                # Load Manning's roughness
                try:
                        with rasterio.open(host_paths["roughness"]) as src_rough:
                                roughness = src_rough.read(1).astype(np.float32)
                                roughness = np.clip(roughness, 0.01, 0.15)
                except Exception:
                        roughness = np.full_like(dem, 0.05, dtype=np.float32)
                        print("    [SIMULATOR] Using default roughness n=0.05")

                # Load hydraulic conductivity (Ksat) if available
                ksat_mmhr = None
                try:
                        ksat_path = host_paths.get("ksat")
                        if ksat_path:
                                with rasterio.open(ksat_path) as src_ksat:
                                        ksat_mmhr = src_ksat.read(1).astype(np.float32)
                except Exception:
                        pass

                if ksat_mmhr is None:
                        # Default values based on land cover (rough estimate)
                        # Urban/impervious: 5 mm/hr, Natural: 25 mm/hr
                        ksat_mmhr = np.full_like(dem, 15.0, dtype=np.float32)
                        print("    [SIMULATOR] Using default Ksat=15 mm/hr")

                # Convert Ksat from mm/hr to m/s
                ksat_ms = ksat_mmhr / (1000.0 * 3600.0)

                # Load imperviousness if available
                imperviousness = None
                try:
                        imp_path = host_paths.get("imperviousness")
                        if imp_path:
                                with rasterio.open(imp_path) as src_imp:
                                        imperviousness = (
                                                src_imp.read(1).astype(np.float32)
                                                / 100.0
                                        )  # Convert to fraction
                except Exception:
                        pass

                if imperviousness is None:
                        # Estimate from roughness: low roughness = urban = high imperviousness
                        # n=0.015 (concrete) -> 80% impervious
                        # n=0.10 (forest) -> 10% impervious
                        imperviousness = np.clip(
                                1.0 - (roughness - 0.015) / 0.085, 0.1, 0.8
                        ).astype(np.float32)
                        print("    [SIMULATOR] Estimated imperviousness from roughness")

                # Valid mask
                valid_mask = (dem > nodata_val) & (dem < 1e10) & np.isfinite(dem)

                # ==================== PHASE 2: INITIALIZE WATER DEPTH ====================

                # Convert precipitation from mm/hr to initial water depth (m)
                # Assume precipitation accumulates over the simulation duration
                total_sim_hours = SIMULATION_DURATION_SECONDS / 3600.0
                initial_water_depth = (
                        precip_mmhr * total_sim_hours
                ) / 1000.0  # Convert to meters

                # Separate into impervious runoff and pervious infiltration
                # Impervious surfaces generate immediate runoff
                impervious_runoff = initial_water_depth * imperviousness

                # Pervious surfaces: some infiltrates, rest becomes runoff
                pervious_precip = initial_water_depth * (1.0 - imperviousness)

                # Maximum infiltration capacity over simulation period
                max_infiltration = ksat_ms * SIMULATION_DURATION_SECONDS  # meters

                # Actual infiltration (limited by capacity)
                infiltration = np.minimum(pervious_precip, max_infiltration)

                # Pervious runoff (excess that can't infiltrate)
                pervious_runoff = pervious_precip - infiltration

                # Total initial runoff available for flooding
                h_initial = impervious_runoff + pervious_runoff
                h_initial = np.where(valid_mask, h_initial, 0.0).astype(np.float32)

                print(
                        f"    [SIMULATOR] Initial water: mean={np.mean(h_initial[valid_mask]):.4f}m, "
                        f"max={np.max(h_initial[valid_mask]):.4f}m"
                )

                # ==================== PHASE 3: ITERATIVE FLOW ROUTING ====================

                h_current = h_initial.copy()
                dt = TIME_STEP_SECONDS
                dx = CELL_SIZE

                # Determine number of iterations based on maximum water depth
                # More water = more iterations needed for stability
                max_depth = np.max(h_initial[valid_mask])
                if max_depth > 0.5:
                        num_iterations = MAX_ITERATIONS
                elif max_depth > 0.1:
                        num_iterations = MAX_ITERATIONS // 2
                else:
                        num_iterations = MAX_ITERATIONS // 4

                num_iterations = max(
                        10, min(num_iterations, MAX_ITERATIONS)
                )  # Clamp to reasonable range

                print(
                        f"    [SIMULATOR] Running {num_iterations} iterations (dt={dt}s, total={num_iterations * dt / 60:.1f}min)"
                )

                # Main simulation loop
                for iteration in range(num_iterations):
                        # Calculate flow using Manning's equation
                        h_after_flow = calculate_manning_flow(
                                h_current, dem, roughness, valid_mask, dt, dx, ny, nx
                        )

                        # Apply infiltration losses (small continuous loss)
                        # Scale Ksat by time step
                        ksat_scaled = (
                                ksat_ms * (dt / SIMULATION_DURATION_SECONDS) * 0.1
                        )  # Small incremental loss
                        imperv_scaled = imperviousness.astype(np.float32)

                        h_current = calculate_infiltration(
                                h_after_flow,
                                ksat_scaled,
                                imperv_scaled,
                                valid_mask,
                                dt,
                                ny,
                                nx,
                        )

                        # Progress indicator every 10%
                        if (iteration + 1) % max(1, num_iterations // 10) == 0:
                                progress = (iteration + 1) / num_iterations * 100
                                current_max = np.max(h_current[valid_mask])
                                print(
                                        f"    [SIMULATOR] Progress: {progress:.0f}% | Max depth: {current_max:.3f}m"
                                )

                # ==================== PHASE 4: POST-PROCESSING ====================

                final_depth = h_current.copy()

                # Apply realistic constraints
                # Remove very small depths (< 1mm)
                final_depth[final_depth < 0.001] = 0.0

                # Apply elevation-based adjustment (high areas drain better)
                dem_normalized = (dem - np.min(dem[valid_mask])) / (
                        np.max(dem[valid_mask]) - np.min(dem[valid_mask]) + 1e-6
                )
                high_elevation_factor = np.where(
                        dem_normalized > 0.8, 0.5, 1.0
                )  # 50% reduction for top 20% elevation
                final_depth = final_depth * high_elevation_factor

                # Slope-based drainage (steeper slopes drain faster)
                grad_y, grad_x = np.gradient(dem)
                slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                slope_normalized = np.clip(
                        slope_magnitude
                        / np.percentile(slope_magnitude[valid_mask], 95),
                        0,
                        1,
                )
                drainage_factor = 1.0 - (
                        slope_normalized * 0.3
                )  # Up to 30% reduction on steep slopes
                final_depth = final_depth * drainage_factor

                # Apply nodata mask
                final_depth[~valid_mask] = -9999.0

                # Ensure realistic range
                final_depth = np.clip(final_depth, 0, 10.0)
                final_depth[~valid_mask] = -9999.0

                # ==================== PHASE 5: SAVE OUTPUT ====================

                profile.update(dtype="float32", nodata=-9999.0, compress="lzw")
                with rasterio.open(host_paths["output_raw"], "w", **profile) as dst:
                        dst.write(final_depth.astype(np.float32), 1)

                # Print statistics
                valid_depth = final_depth[valid_mask & (final_depth > 0)]
                if len(valid_depth) > 0:
                        print("    [SIMULATOR] Final flood statistics:")
                        print(
                                f"                Min: {valid_depth.min():.3f}m | "
                                f"Max: {valid_depth.max():.3f}m | "
                                f"Mean: {valid_depth.mean():.3f}m"
                        )
                        print(
                                f"                Flooded area: {len(valid_depth)} pixels "
                                f"({len(valid_depth) / np.sum(valid_mask) * 100:.1f}% of valid area)"
                        )
                else:
                        print("    [SIMULATOR] No flooding detected")

                return True

        except Exception as e:
                print(f"    [SIMULATOR] Error during simulation: {e}")
                import traceback

                traceback.print_exc()
                return False
