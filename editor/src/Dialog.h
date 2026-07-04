#pragma once
#include <string>
#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#endif

namespace ed {

inline std::string openFileDialog(const char* filter = "MASS project (*.mass)\0*.mass\0All\0*.*\0") {
#ifdef _WIN32
    char file[MAX_PATH] = {0};
    OPENFILENAMEA ofn = {0};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn)) return file;
#endif
    return "";
}

inline std::string saveFileDialog(const char* defExt = "mass",
                                   const char* filter = "MASS project (*.mass)\0*.mass\0All\0*.*\0") {
#ifdef _WIN32
    char file[MAX_PATH] = {0};
    OPENFILENAMEA ofn = {0};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = defExt;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetSaveFileNameA(&ofn)) return file;
#endif
    return "";
}

inline std::string pickFolderDialog() {
#ifdef _WIN32
    // simple folder pick via save dialog trick is clumsy; use SHBrowseForFolder
    BROWSEINFOA bi = {0};
    bi.lpszTitle = "Select the output folder";
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    LPITEMIDLIST pidl = SHBrowseForFolderA(&bi);
    if (pidl) {
        char path[MAX_PATH];
        if (SHGetPathFromIDListA(pidl, path)) return path;
    }
#endif
    return "";
}

} // namespace ed
