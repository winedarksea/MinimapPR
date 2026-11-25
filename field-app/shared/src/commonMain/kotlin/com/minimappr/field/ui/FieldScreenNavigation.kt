package com.minimappr.field.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag

enum class FieldScreen(val title: String) {
    CAPTURE("Capture"),
    CHIRP_LAB("Chirp lab"),
    SESSION("Session"),
}

/**
 * Tab bar over the three service modes. Plain state, no navigation library:
 * three screens, no back stack worth keeping. `initialScreen` follows the
 * running mode so an `am start --es mode ...` launch opens on its screen.
 */
@Composable
fun FieldScreenNavigation(initialScreen: FieldScreen, content: @Composable (FieldScreen) -> Unit) {
    var selected by remember(initialScreen) { mutableStateOf(initialScreen) }
    Column(verticalArrangement = Arrangement.spacedBy(ROW_SPACING)) {
        TabRow(selectedTabIndex = selected.ordinal) {
            FieldScreen.entries.forEach { screen ->
                Tab(
                    selected = screen == selected,
                    onClick = { selected = screen },
                    text = { Text(screen.title) },
                    modifier = Modifier.testTag("field-tab-${screen.name.lowercase()}"),
                )
            }
        }
        content(selected)
    }
}
