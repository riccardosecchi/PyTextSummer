#!/usr/bin/env python3
"""
PyTextSummer Build Script
=========================
Automates the creation of macOS .dmg installer using PyInstaller.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> bool:
    """Run a shell command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main build process."""
    print("\n🚀 PyTextSummer Build Process Starting...\n")
    
    # Get project root
    project_dir = Path(__file__).parent.resolve()
    os.chdir(project_dir)
    
    # 1. Clean previous builds
    print("\n🧹 Cleaning previous builds...")
    for path in ["build", "dist", "__pycache__"]:
        if Path(path).exists():
            shutil.rmtree(path)
            print(f"   Removed: {path}/")
    
    # 2. Check venv activation
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("\n⚠️  Warning: Virtual environment not activated!")
        print("   Please run: source venv/bin/activate")
        sys.exit(1)
    
    # 3. Install/upgrade build dependencies
    print("\n📦 Installing build dependencies...")
    if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pyinstaller"]):
        print("❌ Failed to install PyInstaller")
        sys.exit(1)
    
    # 4. Run PyInstaller
    print("\n🔨 Building macOS app bundle with PyInstaller...")
    pyinstaller_cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=PyTextSummer",
        "--windowed",
        "--onedir",
        "--clean",
        "--noconfirm",
        "--osx-bundle-identifier=com.riccardosecchi.pytextsummer",
        "gemini_latex_gui.py"
    ]
    
    if not run_command(pyinstaller_cmd):
        print("❌ PyInstaller build failed")
        sys.exit(1)
    
    # 5. Verify .app was created
    app_path = project_dir / "dist" / "PyTextSummer.app"
    if not app_path.exists():
        print(f"❌ App bundle not found at: {app_path}")
        sys.exit(1)
    
    print(f"\n✅ App bundle created: {app_path}")
    
    # 6. Create DMG
    print("\n📀 Creating DMG installer...")
    
    # Check if create-dmg is installed
    try:
        subprocess.run(["create-dmg", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n📥 Installing create-dmg...")
        if not run_command(["brew", "install", "create-dmg"]):
            print("⚠️  Could not install create-dmg via Homebrew")
            print("   You can create the DMG manually or install create-dmg first")
            print(f"   App is ready at: {app_path}")
            return
    
    # Create DMG
    dmg_name = "PyTextSummer.dmg"
    if Path(dmg_name).exists():
        Path(dmg_name).unlink()
    
    create_dmg_cmd = [
        "create-dmg",
        "--volname", "PyTextSummer",
        "--window-pos", "200", "120",
        "--window-size", "800", "400",
        "--icon-size", "100",
        "--app-drop-link", "600", "185",
        dmg_name,
        "dist/PyTextSummer.app"
    ]
    
    if run_command(create_dmg_cmd):
        print(f"\n✅ DMG created successfully: {project_dir / dmg_name}")
        print(f"   Size: {Path(dmg_name).stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("\n⚠️  DMG creation had issues, but app bundle is ready")
        print(f"   You can still use: {app_path}")
    
    # 7. Summary
    print("\n" + "="*60)
    print("🎉 BUILD COMPLETE!")
    print("="*60)
    print(f"\n📦 App Bundle: {app_path}")
    if Path(dmg_name).exists():
        print(f"💿 DMG Installer: {project_dir / dmg_name}")
        print(f"\n🚀 Next steps:")
        print(f"   1. Test the app: open {app_path}")
        print(f"   2. Test the DMG: open {dmg_name}")
        print(f"   3. Create GitHub release: gh release create v1.0.0 {dmg_name}")
    print()


if __name__ == "__main__":
    main()
