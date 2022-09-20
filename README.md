# geocfd

Process geometry for cfd simulation.

`Now`:

Read two adjacent buildings, process repeated vertices and build two `nef polyhedra`.

`To do`:

Union two nef polyhedra and output as one building via `.cityjson` file.

# How to use?

The files are built and executed via `WSL` on a `windows10` platform.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

If you have `WSL` and `vscode`(with C++ extension installed on WSL), just clone this project and open it in `vscode`, you should be able to build and run:

after `build`, do:

```console
cd build (this command means you enter into geocfd/build folder)
./geocfd (this command means you execute the geocfd project)
```
for example:

-> build

![image](https://user-images.githubusercontent.com/72781910/191267077-34bac47c-954f-4e0e-9397-194cae06594c.png)

-> execute(in the terminal)

![image](https://user-images.githubusercontent.com/72781910/191267218-2a77ef4e-a575-4288-9ce4-69a2f412709d.png)

Then you could see some prompt information in the terminal:

![image](https://user-images.githubusercontent.com/72781910/191267583-f2908ce0-d295-4285-8e01-ae2ef9864346.png)

-------------------------------------------------------------------------------------------------------------------------------------------------------------
If you use other platforms (such as 'windows' or `MacOS`), you can refer to `CMakeLists.txt` file and use it to build a `CMake` project using `src`, `include` and `data` folder.
