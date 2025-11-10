import rasterio
import numpy as np


def demo_simulation(host_paths):
        """
        Simulate flood inundation based on DEM, precipitation, and Manning's roughness, using basic hydrological principles:
        - Lower elevation areas accumulate water more easily
        - Higher precipitation causes deeper flooding
        - Manning's roughness affects drainage capacity
        - Terrain slope affects flow dynamicsd
        """
        try:
                # Read DEM
                with rasterio.open(host_paths["dem"]) as src:
                        profile = src.profile
                        dem = src.read(1).astype(np.float32)
                        nodata_val = src.nodata if src.nodata is not None else -9999.0

                # Read precipitation data
                with rasterio.open(host_paths["rain"]) as src_rain:
                        precip = src_rain.read(1).astype(np.float32)

                # Read Manning's roughness
                try:
                        with rasterio.open(host_paths["roughness"]) as src_rough:
                                roughness = src_rough.read(1).astype(np.float32)
                                # Normalize roughness (0.01-0.15 typical range)
                                roughness = np.clip(roughness, 0.01, 0.15)
                except Exception:
                        # If roughness not available, use default value
                        roughness = np.full_like(dem, 0.05)

                # Mask for nodata
                valid_mask = (dem > nodata_val) & (dem < 1e10)

                # 1. Calculate terrain slope (gradient) - flat areas accumulate water more
                # Use numpy gradient to calculate slope
                dem_filled = np.where(valid_mask, dem, np.nan)

                # Calculate gradient using numpy (simple finite difference)
                grad_y, grad_x = np.gradient(dem_filled)
                grad_y = np.nan_to_num(grad_y, nan=0.0)
                grad_x = np.nan_to_num(grad_x, nan=0.0)
                slope = np.sqrt(grad_x**2 + grad_y**2)

                # Normalize slope (0-1, where 0 = flat)
                slope_norm = np.clip(
                        slope / (np.nanpercentile(slope[valid_mask], 95) + 1e-6), 0, 1
                )

                # 2. Calculate relative elevation (lower areas flood more easily)
                dem_min = np.nanmin(dem[valid_mask])
                dem_max = np.nanmax(dem[valid_mask])
                dem_range = dem_max - dem_min + 1e-6
                # Elevation factor: 0 = highest, 1 = lowest
                elevation_factor = 1.0 - (dem - dem_min) / dem_range
                elevation_factor = np.clip(elevation_factor, 0, 1)

                # 3. Calculate retention capacity (water retention ability)
                # Flat areas + high roughness = good water retention
                retention = (1.0 - slope_norm) * (roughness / 0.15)
                retention = np.clip(retention, 0, 1)

                # 4. Calculate runoff coefficient
                # High slope + low roughness = fast runoff, less flooding
                runoff_coeff = slope_norm * (1.0 - roughness / 0.15) * 0.7
                runoff_coeff = np.clip(runoff_coeff, 0, 1)

                # 5. Calculate effective precipitation (precipitation causing flooding)
                # precip [mm/hr] -> convert to meters
                precip_m = precip / 1000.0
                # Subtract runoff portion
                effective_precip = precip_m * (1.0 - runoff_coeff)

                # 6. Calculate flood depth based on multiple factors
                # depth = effective_precip * elevation_factor * retention * intensity_factor
                # Intensity factor: higher rainfall increases flooding non-linearly
                intensity_factor = np.where(
                        precip > 0,
                        1.0
                        + np.log1p(
                                precip / 10.0
                        ),  # Non-linear increase with heavy rain
                        0.0,
                )

                # Flood depth calculation (in meters)
                raw_depth = (
                        effective_precip
                        * elevation_factor
                        * retention
                        * intensity_factor
                        * 5.0  # Scaling factor for realistic depth (0-5m range)
                )

                # 7. Apply physical constraints
                # Flooding cannot exceed a threshold based on rainfall amount
                max_possible_depth = precip_m * 3.0  # Assume max 3x precipitation
                raw_depth = np.minimum(raw_depth, max_possible_depth)

                # No flooding if no rain
                raw_depth[precip <= 0] = 0.0

                # High elevation areas (top 20%) flood less
                high_elevation_threshold = np.nanpercentile(dem[valid_mask], 80)
                high_areas = dem > high_elevation_threshold
                raw_depth[high_areas] *= 0.3  # Reduce flood depth by 70% in high areas

                # Apply nodata mask
                raw_depth[~valid_mask] = -9999.0

                # Ensure realistic range (0-10m)
                raw_depth = np.clip(raw_depth, 0, 10.0)
                raw_depth[~valid_mask] = -9999.0

                # 8. Save output
                profile.update(dtype="float32", nodata=-9999.0, compress="lzw")
                with rasterio.open(host_paths["output_raw"], "w", **profile) as dst:
                        dst.write(raw_depth.astype(np.float32), 1)

                # Print statistics
                valid_depth = raw_depth[valid_mask & (raw_depth > 0)]
                if len(valid_depth) > 0:
                        print(
                                f"    [DEMO] Flood statistics: min={valid_depth.min():.3f}m, "
                                f"max={valid_depth.max():.3f}m, mean={valid_depth.mean():.3f}m"
                        )
                        print(
                                f"    [DEMO] Flooded area: {len(valid_depth)} pixels "
                                f"({len(valid_depth) / np.sum(valid_mask) * 100:.1f}% of valid area)"
                        )

                return True

        except Exception as e:
                print(f"    [DEMO] Error creating simulation file: {e}")
                import traceback

                traceback.print_exc()
                return False
