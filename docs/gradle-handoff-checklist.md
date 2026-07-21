# Gradle Project Handoff Checklist

Use this checklist when bringing the Gradle project from the home computer into this repository.

## Files To Bring

- `settings.gradle` or `settings.gradle.kts`
- Root `build.gradle` or `build.gradle.kts`
- `gradle.properties`, if it exists
- `gradlew`
- `gradlew.bat`
- `gradle/wrapper/gradle-wrapper.properties`
- `gradle/wrapper/gradle-wrapper.jar`
- App or module folders, such as `app/`, `android/`, or `WeldVisionApp/`
- Source files under paths like `app/src/main/`
- `AndroidManifest.xml`, if this is an Android project

## Do Not Commit

- `.gradle/`
- `build/`
- `app/build/`
- `.idea/`, unless the file is intentionally shared project config
- `local.properties`
- Generated APK/AAB files
- Machine-specific SDK paths

## After Copying Into This Repo

1. Run `git status --short` and check that only source/config files are new.
2. Confirm the Gradle wrapper exists with `gradlew.bat`.
3. Check whether `local.properties` was copied by mistake.
4. Run a Gradle task such as `gradlew.bat tasks` or `gradlew.bat build`.
5. If the project needs Android SDK paths, keep them local and out of Git.
6. Document how the Gradle project connects to the C++/Python WeldVision work.

## Notes

The current repository does not contain Gradle files yet. As of 2026-07-21, the visible Phase 2 implementation is the Python/Gradio demo under `phase2/`.
