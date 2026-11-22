import gobley.gradle.GobleyHost
import gobley.gradle.Variant
import gobley.gradle.cargo.dsl.jvm
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.library)
    alias(libs.plugins.compose.compiler)
    alias(libs.plugins.compose.multiplatform)
    alias(libs.plugins.kotlin.atomicfu)
    alias(libs.plugins.gobley.cargo)
    alias(libs.plugins.gobley.uniffi)
}

// iOS frameworks need full Xcode (SDK + linker); Command Line Tools alone
// cannot link them. Declare the targets only where they can build, so the
// Android build stays green on CLT-only hosts.
val fullXcodeIsAvailable: Boolean = GobleyHost.Platform.MacOS.isCurrent && runCatching {
    providers.exec {
        commandLine("xcrun", "--sdk", "iphonesimulator", "--show-sdk-path")
        isIgnoreExitValue = true
    }.result.get().exitValue == 0
}.getOrDefault(false)

val rustVariant = Variant(providers.gradleProperty("fieldapp.rust.variant").getOrElse("debug"))

cargo {
    // The FFI crate is a member of the repo-root Cargo workspace.
    packageDirectory = layout.projectDirectory.dir("../../minimappr-ffi")
    jvmVariant = rustVariant
    nativeVariant = rustVariant
    builds.jvm {
        // The host-JVM library only serves unit tests: build for this host,
        // not every desktop triple Gobley would otherwise attempt.
        embedRustLibrary = (rustTarget == GobleyHost.current.rustTarget)
    }
}

uniffi {
    generateFromLibrary {
        namespace = "minimappr_ffi"
        packageName = "com.minimappr.field.ffi"
        variant = rustVariant
    }
}

kotlin {
    androidTarget {
        compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
    }
    // Host JVM target: runs commonTest against the real Rust library.
    jvm {
        compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
    }
    if (fullXcodeIsAvailable) {
        listOf(iosArm64(), iosSimulatorArm64()).forEach { iosTarget ->
            iosTarget.binaries.framework {
                baseName = "FieldShared"
                isStatic = true
            }
        }
    } else {
        logger.lifecycle("field-app: full Xcode not found; iOS targets skipped on this host")
    }

    sourceSets {
        commonMain.dependencies {
            implementation(compose.runtime)
            implementation(compose.foundation)
            implementation(compose.material3)
            implementation(compose.ui)
            implementation(libs.kotlinx.coroutines.core)
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
            implementation(libs.kotlinx.coroutines.test)
        }
        androidMain.dependencies {
            implementation(libs.kotlinx.coroutines.android)
            implementation(libs.androidx.core.ktx)
        }
    }
}

android {
    namespace = "com.minimappr.field.shared"
    compileSdk = libs.versions.android.compileSdk.get().toInt()
    ndkVersion = libs.versions.android.ndk.get()
    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
        // arm64 for phones, x86_64 for emulators; armv7/x86 phones are out
        // of scope for Field Mode (no 32-bit-only devices with raw capture).
        ndk.abiFilters += setOf("arm64-v8a", "x86_64")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
