# include <Mouse.h>  // HID Mouse Library for Arduino
#
# void setup() {
# // UART(TX1 / RX1) - PC Connection
#     Serial1.begin(115200);
#     Serial.begin(115200); // USB CDC initialization
#     Serial.println("Leonardo USB and UART communication initialized.");
#     Mouse.begin();
# }
#
# void loop()
# {
# // 1.
# PC → Arduino(UART)
# Data
# logging
# if (Serial1.available() > 0)
# {
#
#     char
# received = Serial1.read(); // Read
# siganl
# Serial.print("Received from PC: ");
# Serial.println(received);
# // Mouse.click(MOUSE_LEFT); // Mouse
# click
#
# if (received == 'T')
# {
#     Serial.println("Sending Left Click Signal");
# // Mouse.move(50, 50);
# Mouse.click(); // Mouse
# click
# Serial.println("Sent Left Click Signal");
# }
# }}