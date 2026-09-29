"""Diagnostic entry point: ``python -m thermalcam.probe``

Run this first on any new machine. It reports the environment, scans camera
indices, and for whichever one is the thermal camera prints a decoded frame
summary -- so you find out immediately whether raw capture works, instead of
discovering it later through wrong temperatures.

    python -m thermalcam.probe                 # scan and validate
    python -m thermalcam.probe -i 1            # test one index
    python -m thermalcam.probe --debug         # show OpenCV's own log output
    python -m thermalcam.probe --raw 1         # dump what each path returns
    python -m thermalcam.probe --save-config   # write thermalcam.json
"""

from __future__ import annotations

import argparse
import platform
import sys


def _hr(title: str = "") -> None:
    print("-" * 62 if not title else f"--- {title} " + "-" * (57 - len(title)))


def _describe_foreign(frame) -> str:
    """Say what a non-thermal camera looks like, in plain terms.

    Reporting a decode failure ("unexpected row stride: 400 bytes/row") is
    accurate and useless. What the reader needs to know is that this index is a
    different camera altogether -- almost always the built-in webcam -- so they
    stop trying to fix a driver and go look at the cable.
    """
    from .decode import FRAME_HEIGHT, SENSOR_WIDTH

    expected = SENSOR_WIDTH * FRAME_HEIGHT * 2
    if len(frame.shape) == 3:
        return (f"BAD  a different camera: {frame.shape[1]}x{frame.shape[0]}, "
                f"not the TC001's 256x384")
    size = int(frame.size)
    return (f"BAD  a different camera: {size} bytes/frame, "
            f"the TC001 sends {expected}")


def _verdict(frame) -> str:
    """Actually decode a captured frame and say whether it is usable.

    Judging by frame shape alone is not good enough: the raw buffer arrives in
    several different shapes depending on the backend. MSMF, for instance,
    returns the whole untouched payload as a single flat row of 196608 bytes,
    which looks nothing like an image but decodes perfectly. So this runs the
    real decode path and reports the temperature it got.
    """
    import numpy as np

    from .camera import has_thermal_geometry
    from .decode import (FrameLayoutError, detect_byte_order, normalize_buffer,
                         raw_from_thermal, raw_to_celsius,
                         scene_rejection_reason, split_halves)

    thermal_sized = has_thermal_geometry(tuple(frame.shape))

    try:
        buffer = normalize_buffer(frame)
    except FrameLayoutError as exc:
        if not thermal_sized:
            return _describe_foreign(frame)
        if "converted frame" in str(exc):
            return "BAD  converted to BGR, thermal data lost"
        return f"BAD  {str(exc).split('.')[0][:46]}"

    _, thermal = split_halves(buffer)
    celsius = raw_to_celsius(raw_from_thermal(thermal, detect_byte_order(thermal)))

    rejection = scene_rejection_reason(celsius)
    if rejection is not None:
        if not thermal_sized:
            return _describe_foreign(frame)
        return f"BAD  {rejection}"

    median = float(np.median(celsius))
    return (f"GOOD median {median:.1f}C  "
            f"range {celsius.min():.1f}..{celsius.max():.1f}C")


def _list_devices() -> int:
    """Print the cameras the operating system reports, with names.

    This is the question OpenCV cannot answer. When a camera is visible in
    Device Manager but the index scan finds nothing, this shows whether it
    reached the capture subsystem at all.
    """
    import sys as _sys

    from .devices import list_video_devices

    devices = list_video_devices()
    print("Video devices reported by the system")
    if not devices:
        print("  (could not enumerate any)")
        if _sys.platform.startswith("win"):
            print("  For names and indices on Windows:  pip install pygrabber")
        return 1

    for device in devices:
        if device.looks_thermal:
            flag = "  <-- looks like the thermal camera"
        elif device.is_hello_ir:
            flag = "  <-- Windows Hello face unlock, NOT a thermal camera"
        elif not device.addressable:
            flag = "  (no index; not visible to the capture layer)"
        else:
            flag = ""
        print(f"  {device}{flag}")

    openable = [d for d in devices if d.addressable]
    print()
    print(f"  {len(openable)} of {len(devices)} can be opened by OpenCV.")

    if not any(d.looks_thermal and d.addressable for d in devices):
        print()
        if openable:
            print("  No device name identifies a thermal camera.")
            print("  Generic USB/UVC names need frame validation; run:")
            print("      python -m thermalcam.probe")
        else:
            print("  No addressable video camera was found.")
            if _sys.platform.startswith("win"):
                print("  To check whether Windows sees the hardware, run:")
                print("      python -m thermalcam.probe --detect")

    return 0


def _detect_by_unplugging() -> int:
    """Find the camera's device node by watching what disappears.

    When a camera is invisible to the capture layer, the decisive question is
    whether Windows notices the hardware at all. Comparing the full device list
    with the camera plugged in against the same list without it answers that
    with no guesswork: whatever vanished is the camera, and its problem code
    says why it is not usable. If nothing vanishes, the device never reached
    the operating system, and no amount of software will help.
    """
    import sys as _sys

    from .devices import diff_snapshots, snapshot_pnp_devices

    if not _sys.platform.startswith("win"):
        print("This check is Windows-only.")
        print("On Linux, run:  dmesg -w   and watch while you plug the camera in.")
        return 1

    print("Hardware detection")
    print("-" * 62)
    print("This finds the camera by seeing which device node disappears when")
    print("you unplug it. Nothing is changed on your system.")
    print()
    input("  1. Plug the THERMAL camera in, wait a few seconds, "
          "then press Enter... ")
    before = snapshot_pnp_devices()
    if not before:
        print("\nCould not read the device list. Is PowerShell available?")
        return 1
    print(f"     {len(before)} devices present.")
    print()
    input("  2. Now UNPLUG the thermal camera, wait a few seconds, "
          "then press Enter... ")
    after = snapshot_pnp_devices()
    print(f"     {len(after)} devices present.")
    print()

    vanished = diff_snapshots(before, after)
    print("-" * 62)

    if not vanished:
        print("NOTHING disappeared when you unplugged the camera.")
        print()
        print("Windows never registered the device, so this is not a driver,")
        print("OpenCV or Python problem -- the hardware is not making contact.")
        print()
        print("  - Try the other orientation of the USB-C plug. Flip it over.")
        print("    Badly-wired cables work in one orientation only.")
        print("  - Use the exact cable that works on the other machine.")
        print("  - Try every USB port, including USB-A with the adapter.")
        print("  - Unplug the external webcam first, in case the port or hub")
        print("    is out of power budget.")
        print("  - Plug directly into the laptop, never through a hub, dock")
        print("    or monitor.")
        return 1

    print(f"Found {len(vanished)} device node(s) belonging to the camera:")
    print()
    healthy = []
    for entry in vanished:
        mark = "OK  " if entry.is_healthy else "FAIL"
        print(f"  [{mark}] {entry.name}")
        print(f"         class: {entry.device_class or '(none)'}   "
              f"status: {entry.status}   {entry.problem_text}")
        print(f"         {entry.instance_id}")
        if entry.is_healthy:
            healthy.append(entry)
    print()

    hijacked = [e for e in vanished if e.hijacked_by]
    cameras = [e for e in vanished if e.is_camera_class]

    if hijacked:
        _explain_hijack(hijacked[0])
    elif cameras and all(e.is_healthy for e in cameras):
        print("Windows sees it as a healthy camera, so the fault is above the")
        print("driver. Close anything that might be holding it (Camera app,")
        print("Teams, Zoom, OBS) and check Settings > Privacy & security >")
        print("Camera > 'Let desktop apps access your camera'.")
    elif cameras:
        print("It registers as a camera but is not healthy -- see the problem")
        print("text above. Uninstall the device in Device Manager, then unplug")
        print("and replug it so Windows reinstalls the generic UVC driver.")
    else:
        print("Windows sees the hardware but has NOT classified it as a camera.")
        print("That is why no capture backend can find it.")
        print()
        print("In Device Manager, find the entry above, right-click it and")
        print("choose 'Update driver' > 'Browse my computer' > 'Let me pick'")
        print("> 'USB Video Device'. If the TOPDON PC software is installed,")
        print("uninstall it first -- its driver claims the device and prevents")
        print("the generic UVC one from binding.")
    return 0


def _explain_hijack(entry) -> None:
    """Spell out how to hand a device back to the camera stack."""
    driver = entry.hijacked_by
    print(f"THIS IS THE PROBLEM: the camera is bound to {driver}, not to")
    print("Windows' USB Video Class driver.")
    print()
    print(f"{driver} is a generic USB driver, normally installed by a tool")
    print("called Zadig so that user-space software can talk to a device")
    print("directly. Something on this machine did that to your camera -- most")
    print("likely while setting up unrelated hardware such as an SDR, a")
    print("microcontroller programmer or a flashing tool.")
    print()
    print("The device reports 'working normally' because it genuinely is,")
    print("as a generic USB device. It is simply no longer a camera, which is")
    print("why nothing in the capture stack can see it. This is also why the")
    print("same camera and cable work on the other machine.")
    print()
    print("To hand it back, in Device Manager (you will need an admin account):")
    print()
    print("  1. View > Show hidden devices.")
    print(f"  2. Expand '{entry.device_class}' and find:")
    print(f"       {entry.name}")
    print("  3. Right-click it > Uninstall device.")
    print("  4. TICK 'Attempt to remove the driver for this device'.")
    print("     Skipping this box is why the problem usually comes back.")
    print("  5. Unplug the camera, wait five seconds, plug it back in.")
    print()
    print("It should reappear under 'Cameras' as a USB video device. Confirm:")
    print("      python -m thermalcam.probe --list")
    print()
    print(f"If it binds to {driver} again, the driver package is still")
    print("installed. In an ADMINISTRATOR PowerShell:")
    print()
    print("      pnputil /enum-drivers")
    print()
    print(f"Find the entry whose provider or class mentions {driver}, note its")
    print("'Published Name' (oemNN.inf), then:")
    print()
    print("      pnputil /delete-driver oemNN.inf /uninstall /force")
    print()
    print("Then unplug and replug the camera. Only remove a driver you are")
    print("sure is the generic USB one -- other devices may depend on it, and")
    print("whatever it was originally installed for will stop working.")


def _dump_raw(index: int, verbose: bool) -> int:
    """Show what every backend/strategy combination hands back, and decode it.

    This is the report to send if the camera still will not open.
    """
    import cv2

    from .camera import (DEFAULT_FPS, RELEASE_SETTLE_S, _backend_name,
                         _platform_backends, set_opencv_log_level)
    from .capture_config import STRATEGIES, convert_rgb_readback
    from .decode import FRAME_HEIGHT, SENSOR_WIDTH

    import time

    set_opencv_log_level(verbose)
    print(f"Raw capture dump for index {index}")
    print(f"{'backend':<11} {'strategy':<14} {'shape':<22} {'RGB':<4} verdict")
    _hr()

    working: list[tuple[str, str]] = []
    foreign = converted_thermal = False
    for backend in _platform_backends():
        for strat in STRATEGIES:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                print(f"{_backend_name(backend):<11} {'-':<14} "
                      f"{'(would not open)':<22}")
                break
            try:
                strat.apply(cap, SENSOR_WIDTH, FRAME_HEIGHT, DEFAULT_FPS)
                readback = convert_rgb_readback(cap)
                shape, verdict = "-", "no frames arrived"
                for _ in range(8):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        shape = f"{frame.shape} {frame.dtype}"
                        verdict = _verdict(frame)
                        break
                flag = "-" if readback is None else f"{readback:g}"
                print(f"{_backend_name(backend):<11} {strat.name:<14} "
                      f"{shape:<22} {flag:<4} {verdict}")
                if verdict.startswith("GOOD"):
                    working.append((_backend_name(backend), strat.name))
                elif "a different camera" in verdict:
                    foreign = True
                elif "converted to BGR" in verdict:
                    converted_thermal = True
            except cv2.error as exc:
                print(f"{_backend_name(backend):<11} {strat.name:<14} "
                      f"error: {str(exc)[:40]}")
            finally:
                cap.release()
                time.sleep(RELEASE_SETTLE_S)

    print()
    if working:
        print(f"{len(working)} working combination(s). The library will use "
              f"{working[0][0]} / {working[0][1]}.")
        print("Raw capture is fine on this machine -- run the probe without "
              "--raw to confirm end to end.")
        return 0

    if foreign and not converted_thermal:
        # Every backend agreed this is some other camera. Saying so plainly
        # saves the reader from debugging a driver problem they do not have.
        print(f"Index {index} is NOT the thermal camera -- it is a different "
              f"device (a built-in webcam, most likely).")
        print("Every backend agreed on that, so it is not a driver or format "
              "problem at this index.")
        print()
        print("The TC001 is not reaching Windows as a UVC device. Check:")
        print("  1. That it is actually plugged in, and its blue LED is lit.")
        print("  2. The cable. Many USB-C cables carry power only. Use the one "
              "that works on the other machine, straight into a USB-C port --")
        print("     no hub, dock or monitor in between.")
        print("  3. Device Manager, under Cameras / Imaging devices, while you "
              "plug it in. Nothing new appearing means cable, port or power.")
        print("  4. Settings > Privacy & security > Camera: 'Let desktop apps "
              "access your camera' must be on.")
    elif converted_thermal:
        print("The thermal camera is at this index but every path converted the "
              "frame.")
        print("Uninstall the TOPDON PC software if present, close any app using "
              "the camera, then retry.")
    else:
        print("No combination produced decodable thermal data at this index.")
        print("If other indices exist, try them: --raw 1, --raw 2, ...")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m thermalcam.probe",
        description="Detect and validate a TC001-class thermal camera.")
    parser.add_argument("-i", "--index", type=int, default=None,
                        help="Test only this camera index (default: scan).")
    parser.add_argument("-n", "--max-index", type=int, default=10,
                        help="Highest index to scan (default: 10).")
    parser.add_argument("--debug", action="store_true",
                        help="Let OpenCV's own [WARN]/[ERROR] lines through.")
    parser.add_argument("--raw", type=int, metavar="INDEX", default=None,
                        help="Dump what every backend/strategy returns for INDEX.")
    parser.add_argument("--list", action="store_true",
                        help="List the video devices the system reports, by name.")
    parser.add_argument("--detect", action="store_true",
                        help="Find the camera's device node by unplugging it. "
                             "Use when --list does not show it at all.")
    parser.add_argument("--save-config", metavar="PATH", nargs="?",
                        const="thermalcam.json", default=None,
                        help="Write a starter config for the camera that was found.")
    args = parser.parse_args(argv)

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Install with:  pip install -r requirements.txt")
        return 1

    from .camera import ThermalCamera, ThermalCameraError, find_cameras

    print("=" * 62)
    print("  thermalcam probe")
    print("=" * 62)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")
    print(f"  OpenCV  : {cv2.__version__}")
    print(f"  NumPy   : {np.__version__}")
    if not args.debug:
        print("  (OpenCV log output suppressed; re-run with --debug to see it)")
    print()

    if args.detect:
        return _detect_by_unplugging()

    if args.list:
        return _list_devices()

    _list_devices()
    print()

    if args.raw is not None:
        return _dump_raw(args.raw, args.debug)

    if args.index is None:
        print(f"Scanning camera indices 0..{args.max_index - 1} ...")
        found = find_cameras(max_index=args.max_index, verbose=args.debug)
        if not found:
            print("  No camera devices responded at any index.")
            print("  Check the cable and that the camera's blue LED is lit.")
            return 1
        for entry in found:
            mark = "THERMAL" if entry["thermal"] else "       "
            name = f"{entry.get('name') or '?':<24.24}"
            print(f"  [{mark}] index {entry['index']:>2}  {name} "
                  f"{entry['backend']:<11} {entry['note']}")
        thermal = [e for e in found if e["thermal"]]
        if not thermal:
            print()
            if any("thermal camera, but converted" in e["note"] for e in found):
                print("The thermal camera is present but every capture path "
                      "converted the frame.")
                print("Uninstall the TOPDON PC software if it is installed, close "
                      "any app using the camera, then retry.")
            else:
                print("No thermal camera found -- only ordinary camera(s) "
                      "responded, most likely a built-in webcam.")
                print()
                print("The TC001 is not reaching Windows as a UVC device. Check, "
                      "in order:")
                print("  1. The cable. Many USB-C cables are charge-only. Use the "
                      "supplied one, directly into the machine, no hub or dock.")
                print("  2. Device Manager, under Cameras / Imaging devices, while "
                      "you plug it in. Nothing appearing means cable, port or power.")
                print("  3. A warning triangle there means the TOPDON driver is "
                      "installed -- uninstall the TOPDON PC software completely.")
                print("  4. Settings > Privacy & security > Camera: 'Let desktop "
                      "apps access your camera' must be on.")
            print()
            print(f"For the full picture:  python -m thermalcam.probe --raw "
                  f"{found[0]['index']}")
            return 1
        index = thermal[0]["index"]
        print(f"\nUsing index {index}.")
    else:
        index = args.index

    print("\nOpening and decoding ...")
    try:
        with ThermalCamera(index=index, verbose=args.debug) as cam:
            info = cam.info()
            for key in ("index", "backend", "strategy", "fourcc", "width",
                        "height", "fps", "swap_bytes"):
                print(f"  {key:<12}: {info[key]}")

            frame = cam.read()
            stats = frame.stats()
            print("\n  Decoded frame")
            print(f"  {'resolution':<12}: {frame.shape[1]}x{frame.shape[0]}")
            print(f"  {'min':<12}: {stats['min_c']:.2f} C  at "
                  f"({frame.coldest.x}, {frame.coldest.y})")
            print(f"  {'max':<12}: {stats['max_c']:.2f} C  at "
                  f"({frame.hottest.x}, {frame.hottest.y})")
            print(f"  {'mean':<12}: {stats['mean_c']:.2f} C")
            print(f"  {'centre':<12}: {stats['center_c']:.2f} C")

            spread = stats["max_c"] - stats["min_c"]
            print()
            if spread < 0.05:
                print("  WARNING: the scene is perfectly flat. That usually means "
                      "the data is not radiometric after all.")
            elif not -20 <= stats["mean_c"] <= 100:
                print("  WARNING: mean temperature is implausible for an indoor "
                      "scene. Check that nothing is converting the frame.")
            else:
                print("  Looks healthy. Point the camera at your hand -- the max "
                      "should read roughly 30-35 C.")

            if args.save_config:
                from .config import Config
                path = Config(device_index=index,
                              swap_bytes=cam.swap_bytes).save(args.save_config)
                print(f"\n  Wrote starter config to {path.resolve()}")

    except ThermalCameraError as exc:
        print(f"\nFAILED\n{exc}")
        print(f"\nFor the full picture, run:  python -m thermalcam.probe --raw {index}")
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
