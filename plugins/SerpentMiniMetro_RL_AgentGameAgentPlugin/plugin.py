import offshoot


class SerpentMiniMetro_RL_AgentGameAgentPlugin(offshoot.Plugin):
    name = "SerpentMiniMetro_RL_AgentGameAgentPlugin"
    version = "0.1.0"

    plugins = []

    libraries = []

    files = [
        {"path": "serpent_MiniMetro_RL_Agent_game_agent.py", "pluggable": "GameAgent"}
    ]

    config = {
        "frame_handler": "PLAY"
    }

    @classmethod
    def on_install(cls):
        print("\n\n%s was installed successfully!" % cls.__name__)

    @classmethod
    def on_uninstall(cls):
        print("\n\n%s was uninstalled successfully!" % cls.__name__)


if __name__ == "__main__":
    offshoot.executable_hook(SerpentMiniMetro_RL_AgentGameAgentPlugin)
