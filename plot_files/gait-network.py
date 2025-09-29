import sys

sys.path.append("../PlotNeuralNet/")
from pycore.tikzeng import *

####################################################################################################
# footstep filtering
####################################################################################################

offset = 1
cords = [
    (-offset, offset),
    (offset, offset),
    (offset, -offset),
    (-offset, -offset),
]

images_path = __file__.replace("gait-network.py", "images")
input_file_paths = [
    f"{images_path}/RL.png",
    f"{images_path}/FL.png",
    f"{images_path}/FR.png",
    f"{images_path}/RR.png",
]

legs = []
for i, ((x, y), input_file) in enumerate(zip(cords, input_file_paths)):

    z = 0
    label = r"Footstep Cost Map" if i == 0 else ""
    curr_layer = f"input_{i}"
    legs.append(
        to_Conv(
            f"input_{i}",
            5,
            5,
            offset=f"({z}, {x}, {y})",
            height=5,
            depth=5,
            width=1,
            caption=label,
        )
    )
    legs.append(
        to_input(
            input_file,
            to=f"({z+0.2}, {x}, {y-1.02})",
            width=-1,
            height=1,
        )
    )
    prev_layer = curr_layer

    z += 2.5
    label = r"Position\\Filters" if i == 0 else ""
    curr_layer = f"filter_{i}"
    legs.append(
        to_Conv(
            curr_layer,
            5,
            5,
            offset=f"({z}, {x}, {y})",
            height=5,
            depth=5,
            width=1,
            caption=label,
        )
    )
    legs.append(
        to_input(
            input_file[:-4]+"_mask.png",
            to=f"({z+0.2}, {x}, {y-1.02})",
            width=-1,
            height=1,
        )
    )

    legs.append(to_connection(prev_layer, curr_layer))
    prev_layer = curr_layer

    z += 2
    label = r"Argmin" if i == 0 else ""
    curr_layer = f"argmax_{i}"
    legs.append(
        to_Conv(
            curr_layer,
            "",  # type: ignore
            "",  # type: ignore
            offset=f"({z}, {x}, {y})",
            height=1,
            depth=1,
            width=1,
            caption=label,
        )
    )

    legs.append(to_connection(prev_layer, curr_layer))
    prev_layer = curr_layer

plot = [
    to_head(".."),
    to_cor(),
    to_begin(),
    *legs,
]

z += 2  # type: ignore
curr_layer = "footstep_options"
plot.append(
    to_Conv(
        curr_layer,
        7,
        12,  # type: ignore
        offset=f"({z}, 0, 0)",
        height=12,
        depth=7,
        width=1,
        caption="Footstep\\\\Options",
    )
)
plot.extend([to_connection(f"argmax_{i}", curr_layer) for i in range(4)])
prev_layer = curr_layer

####################################################################################################
# footstep encoder
####################################################################################################

z += 2.5
curr_layer = "singulate"
plot.append(
    to_Conv(
        curr_layer,
        7,
        "",  # type: ignore
        offset=f"({z}, 0, 0)",
        height=1,
        depth=7,
        width=1,
        caption="Singulate",
    )
)
plot.extend([to_connection(prev_layer, curr_layer) for i in range(4)])
prev_layer = curr_layer

z += 2.7
curr_layer = "fse_d1"
plot.append(
    to_Conv(
        curr_layer,
        32,
        "",  # type: ignore
        offset=f"({z}, 0, 0)",
        height=1,
        depth=24,
        width=1,
        caption="Dense",
    )
)
plot.append(to_connection(prev_layer, curr_layer))
prev_layer = curr_layer

z += 1.5
curr_layer = "fse_d2"
plot.append(
    to_Conv(
        curr_layer,
        32,
        "",  # type: ignore
        offset=f"({z}, 0, 0)",
        height=1,
        depth=24,
        width=1,
        caption="Dense",
    )
)
plot.append(to_connection(prev_layer, curr_layer))
prev_layer = curr_layer
footstep_encoder_end = curr_layer


####################################################################################################
# state encoder
####################################################################################################


z -= 4.5
low_level = -offset * 5
curr_layer = "state_input"
plot.append(
    to_Conv(
        curr_layer,
        22,
        "",  # type: ignore
        offset=f"({z},{low_level},0)",
        height=1,
        depth=22,
        width=1,
        caption=r"State\\Input",
    ),
)
prev_layer = curr_layer

z += 2
curr_layer = "rse_d1"
plot.append(
    to_Conv(
        curr_layer,
        64,
        "",  # type: ignore
        offset=f"({z}, {low_level}, 0)",
        height=1,
        depth=48,
        width=1,
        caption="Dense",
    )
)
plot.append(to_connection(prev_layer, curr_layer))
prev_layer = curr_layer

z += 1.5
curr_layer = "rse_d2"
plot.append(
    to_Conv(
        curr_layer,
        64,
        "",  # type: ignore
        offset=f"({z}, {low_level}, 0)",
        height=1,
        depth=48,
        width=1,
        caption="Dense",
    )
)
plot.append(to_connection(prev_layer, curr_layer))
prev_layer = curr_layer
state_encoder_end = curr_layer

####################################################################################################
# shared trunk
####################################################################################################

z += 4
st_level = low_level / 2
curr_layer = "st_d1"
plot.append(
    to_Conv(
        curr_layer,
        128,
        "",  # type: ignore
        offset=f"({z}, {st_level}, 0)",
        height=1,
        depth=64,
        width=1,
        caption="Dense",
    )
)
plot.append(to_connection(state_encoder_end, curr_layer))
plot.append(to_connection(footstep_encoder_end, curr_layer))
prev_layer = curr_layer

z += 2
curr_layer = "st_d2"
plot.append(
    to_Conv(
        curr_layer,
        64,
        "",  # type: ignore
        offset=f"({z}, {st_level}, 0)",
        height=1,
        depth=48,
        width=1,
        caption="Dense",
    )
)
plot.append(to_connection(prev_layer, curr_layer))
prev_layer = curr_layer
st_trunk_end = curr_layer

####################################################################################################
# outputs
####################################################################################################
offset = 2

z += 3
curr_layer = "value_head"
plot.append(
    to_Conv(
        curr_layer,
        1,
        "",  # type: ignore
        offset=f"({z},{st_level+offset},0)",
        height=1,
        depth=1,
        width=1,
        caption=r"Value\\Head",
    ),
)
plot.append(to_connection(st_trunk_end, curr_layer))
prev_layer = curr_layer

curr_layer = "duration_head"
plot.append(
    to_Conv(
        curr_layer,
        1,
        "",  # type: ignore
        offset=f"({z},{st_level-offset},0)",
        height=1,
        depth=1,
        width=1,
        caption=r"Duration\\Head",
    ),
)
plot.append(to_connection(st_trunk_end, curr_layer))
prev_layer = curr_layer

####################################################################################################
# dummy
####################################################################################################

z += 3
curr_layer = "dummy1"
plot.append(
    to_Conv(
        curr_layer,
        "",  # type: ignore
        "",  # type: ignore
        offset=f"({z},0,0)",
        height=1,
        depth=1,
        width=1,
        caption="",
    ),
)
prev_layer = curr_layer
z += 1
curr_layer = "dummy2"
plot.append(
    to_Conv(
        curr_layer,
        "",  # type: ignore
        "",  # type: ignore
        offset=f"({z},0,0)",
        height=1,
        depth=1,
        width=1,
        caption="",
    ),
)
plot.append(to_connection(prev_layer, curr_layer))

####################################################################################################
# rest
####################################################################################################

plot.append(to_end())


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(plot, namefile + ".tex")


if __name__ == "__main__":
    main()
