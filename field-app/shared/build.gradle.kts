import gobley.gradle.GobleyHost
import gobley.gradle.Variant
import gobley.gradle.cargo.dsl.android
import gobley.gradle.cargo.dsl.jvm
import gobley.gradle.cargo.profiles.CargoProfile
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

// Android Rust profile, independent of the APK build type: debug-built DSP is
// too slow on a phone (plan finding 10), so a debug APK can carry a release
// Rust library. Falls back to fieldapp.rust.variant when unset.
val androidRustVariantName: String = providers.gradleProperty("fieldapp.rust.android.variant")
    .orElse(providers.gradleProperty("fieldapp.rust.variant"))
    .getOrElse("debug")

// arm64-only by default for device builds; add x86_64 for emulators.
val androidAbis: Set<String> = providers.gradleProperty("fieldapp.android.abis")
    .getOrElse("arm64-v8a,x86_64")
    .split(",").map(String::trim).filter(String::isNotEmpty).toSet()

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
    builds.android {
        if (androidRustVariantName == "release") debug.profile.set(CargoProfile.Release)
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
    // jvmCommon = Android + host JVM: java.io file writers and the Ktor LAN
    // transport, testable on the host JVM over real 127.0.0.1 sockets.
    @OptIn(org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi::class)
    applyDefaultHierarchyTemplate {
        common {
            group("jvmCommon") {
                withAndroidTarget()
                withJvm()
            }
        }
    }
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
        getByName("jvmCommonMain").dependencies {
            implementation(libs.ktor.server.core)
            implementation(libs.ktor.server.cio)
            implementation(libs.ktor.server.websockets)
            implementation(libs.ktor.client.core)
            implementation(libs.ktor.client.websockets)
        }
        androidMain.dependencies {
            implementation(libs.kotlinx.coroutines.android)
            implementation(libs.androidx.core.ktx)
            implementation(libs.ktor.client.okhttp)
        }
        jvmMain.dependencies {
            implementation(libs.ktor.client.cio)
        }
    }
}

android {
    namespace = "com.minimappr.field.shared"
    compileSdk = libs.versions.android.compileSdk.get().toInt()
    ndkVersion = libs.versions.android.ndk.get()
    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
        // arm64 for phones, x86_64 for emulators (fieldapp.android.abis);
        // armv7/x86 phones are out of scope for Field Mode.
        ndk.abiFilters += androidAbis
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    packaging {
        // Ktor/kotlinx jars ship duplicate license and module metadata.
        resources { excludes += setOf("/META-INF/{AL2.0,LGPL2.1}", "/META-INF/INDEX.LIST", "/META-INF/io.netty.versions.properties") }
    }
}
