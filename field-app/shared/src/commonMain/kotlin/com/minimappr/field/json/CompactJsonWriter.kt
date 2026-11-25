package com.minimappr.field.json

/**
 * Minimal JSON object writer for events.jsonl, session.json and dumpsys
 * lines. Hand-rolled rather than kotlinx.serialization: the shapes are
 * script contracts written field by field, and this avoids a compiler plugin
 * for a few dozen keys. Non-finite doubles become null (JSON has no NaN).
 */
class JsonObjectBuilder {
    private val members = mutableListOf<String>()

    fun put(key: String, value: String?) = add(key, value?.let(::quoteJsonString) ?: "null")
    fun put(key: String, value: Long?) = add(key, value?.toString() ?: "null")
    fun put(key: String, value: Int?) = add(key, value?.toString() ?: "null")
    fun put(key: String, value: Double?) = add(key, jsonNumber(value))
    fun put(key: String, value: Float?) = add(key, jsonNumber(value?.toDouble()))
    fun put(key: String, value: Boolean?) = add(key, value?.toString() ?: "null")
    fun putObject(key: String, build: JsonObjectBuilder.() -> Unit) = add(key, jsonObject(build))
    fun putRaw(key: String, rawJson: String) = add(key, rawJson)
    fun putStrings(key: String, values: List<String>) = add(key, values.joinToString(",", "[", "]", transform = ::quoteJsonString))
    fun putNumbers(key: String, values: List<Double>) = add(key, values.joinToString(",", "[", "]") { jsonNumber(it) })
    fun putLongs(key: String, values: List<Long>) = add(key, values.joinToString(",", "[", "]"))
    fun putObjects(key: String, values: List<JsonObjectBuilder.() -> Unit>) =
        add(key, values.joinToString(",", "[", "]") { jsonObject(it) })

    private fun add(key: String, rawValue: String) {
        members += "${quoteJsonString(key)}:$rawValue"
    }

    fun build(): String = members.joinToString(",", "{", "}")
}

fun jsonObject(build: JsonObjectBuilder.() -> Unit): String = JsonObjectBuilder().apply(build).build()

private fun jsonNumber(value: Double?): String =
    if (value == null || value.isNaN() || value.isInfinite()) "null" else value.toString()

private const val FIRST_PRINTABLE_CHARACTER = 0x20
private const val HEX_RADIX = 16
private const val UNICODE_ESCAPE_DIGITS = 4

fun quoteJsonString(value: String): String = buildString(value.length + 2) {
    append('"')
    for (character in value) {
        when (character) {
            '"' -> append("\\\"")
            '\\' -> append("\\\\")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> if (character.code < FIRST_PRINTABLE_CHARACTER) {
                append("\\u").append(character.code.toString(HEX_RADIX).padStart(UNICODE_ESCAPE_DIGITS, '0'))
            } else {
                append(character)
            }
        }
    }
    append('"')
}
