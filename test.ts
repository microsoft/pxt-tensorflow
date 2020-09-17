namespace userconfig {
}

console.addListener((pri, txt) => control.dmesg("C: " + txt.slice(0, -1)))
jacdac.consolePriority = ConsolePriority.Log

control.dmesg("Hello")

pins.A9.digitalWrite(false)

jacdac.deviceNamer.start()
jacdac.tfliteHost.start()

jacdac.start()
