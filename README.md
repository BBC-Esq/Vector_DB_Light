<div align="center">
  <h3><u>Requirements</u></h3>

| [🐍 Python 3.11](https://www.python.org/downloads/release/python-3119/) or [Python 3.12](https://www.python.org/downloads/release/python-31210/) &nbsp;&bull;&nbsp; [📁 Git](https://git-scm.com/downloads) &nbsp;&bull;&nbsp; [📁 Git LFS](https://git-lfs.com/) &nbsp;&bull;&nbsp; [🌐 Pandoc](https://github.com/jgm/pandoc/releases) &nbsp;&bull;&nbsp; [🛠️ Compiler](https://visualstudio.microsoft.com/) |
|---|

The above link downloads Visual Studio as an example.  Make sure to install the required SDKs, however.

> <details>
>   <summary>EXAMPLE error when no compiler installed:</summary>
>   <img src="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/sample_error.png?raw=true">
> </details>
> 
> <details>
>   <summary>EXAMPLE of installing the correct SDKs:</summary>
>   <img src="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/build_tools.png?raw=true">
> </details>

</div>

<a name="installation"></a>
<div align="center"> <h2>Installation</h2></div>
  
### Step 1
Download the ZIP file for the latest "release."  Extract its contents and navigate to the `src` folder.
> [!CAUTION]
> If you simply clone this repository you will get the development version, which might not be stable.
### Step 2
Within the `src` folder, create a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/):
```
python -m venv .
```
### Step 3
Activate the virtual environment:
```
.\Scripts\activate
```
### Step 4
Run the setup script:
   > Only ```Windows``` is supported for now.

```
python setup_windows.py
```
