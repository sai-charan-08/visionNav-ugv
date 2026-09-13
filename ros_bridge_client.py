import socket


class ROSBridgeClient:

    def __init__(
        self,
        host="127.0.0.1",
        port=8765
    ):
        self.host = host
        self.port = port

    def send_decision(self, decision):

        decision = decision.strip()

        try:

            with socket.create_connection(
                (self.host, self.port),
                timeout=1.0
            ) as sock:

                message = decision + "\n"

                sock.sendall(
                    message.encode("utf-8")
                )

            print(
                f"[ROSBridge] Sent decision: {decision}"
            )

            return True

        except OSError as error:

            print(
                f"[ROSBridge] Connection failed: {error}"
            )

            return False

    def stop(self):

        return self.send_decision("STOP")


if __name__ == "__main__":

    client = ROSBridgeClient()

    print("Testing Windows → ROS 2 bridge...")

    client.stop()

    print("Bridge test completed.")