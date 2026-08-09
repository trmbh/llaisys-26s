target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    add_rules("cuda")
    set_policy("build.cuda.devlink", true)
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
    end
    add_files("../src/device/nvidia/*.cu")
    add_cugencodes("native")
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    add_rules("cuda")
    set_policy("build.cuda.devlink", true)
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
    end
    add_files("../src/ops/nvidia/*.cu")
    add_cugencodes("native")
    on_install(function (target) end)
target_end()

