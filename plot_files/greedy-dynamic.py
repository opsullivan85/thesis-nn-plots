import sys

sys.path.append("../PlotNeuralNet/")
from pycore.tikzeng import *

centerline_offset = 2.5

# defined your arch
arch = [
    to_head(".."),
    to_cor(),
    to_begin(),
    to_Conv(
        "terrain_input",
        5,
        5,
        offset=f"(0,-{centerline_offset},0)",
        height=5,
        depth=5,
        width=1,
        caption="Terrain\\\\Input",
    ),
    to_Conv(
        "terrain_conv1",
        3,
        3,
        offset=f"(2,-{centerline_offset},0)",
        height=3,
        depth=16,
        width=3,
        caption="Conv",
    ),
    to_connection("terrain_input", "terrain_conv1"),
    to_Conv(
        "terrain_dense2",
        64,
        "",
        offset=f"(5,-{centerline_offset},0)",
        height=1,
        depth=64,
        width=1,
        caption="Dense",
    ),
    to_connection("terrain_conv1", "terrain_dense2"),
    to_Conv(
        "state_input",
        43,
        "",
        offset=f"(0,{centerline_offset},0)",
        height=1,
        depth=43,
        width=1,
        caption="State\\\\Input",
    ),
    to_Conv(
        "state_dense1",
        64,
        "",
        offset=f"(2,{centerline_offset},0)",
        height=1,
        depth=64,
        width=1,
        caption="Dense",
    ),
    to_connection("state_input", "state_dense1"),
    to_Conv(
        "state_dense2",
        64,
        "",
        offset=f"(5,{centerline_offset},0)",
        height=1,
        depth=64,
        width=1,
        caption="Dense",
    ),
    to_connection("state_dense1", "state_dense2"),
    to_Conv(
        "conn_dense1",
        64,
        "",
        offset=f"(9,0,0)",
        height=1,
        depth=64,
        width=1,
        caption="Dense",
    ),
    to_connection("state_dense2", "conn_dense1"),
    to_connection("terrain_dense2", "conn_dense1"),
    to_Conv(
        "conn_output2",
        5,
        5,
        offset=f"(11, 0,0)",
        height=5,
        depth=5,
        width=1,
        caption="Preference\\\\Output",
    ),
    to_connection("conn_dense1", "conn_output2"),
    to_Conv(
        "test",
        5,
        5,
        offset=f"(15, 0,0)",
        height=5,
        depth=5,
        width=1,
        caption="",
    ),
    to_connection("conn_output2", "test"),
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
