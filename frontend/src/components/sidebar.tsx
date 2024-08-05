import React from "react";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import AddIcon from "@mui/icons-material/Add";
import SettingsIcon from "@mui/icons-material/Settings";

const Sidebar = () => {
  return (
    <div className="flex w-full flex-col bg-white border-r border-gray-200 h-full">
      <div className="flex items-center p-4 border-b border-gray-200">
        <input
          type="text"
          placeholder="Enter Project Name"
          className="border border-teal-500 rounded py-2 px-4 text-gray-600 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 bg-gray-100 flex-grow w-full"
        />
        <AccountTreeIcon className="text-black ml-4" />
      </div>
      <div className="flex items-center justify-between px-3 py-4">
        <h2 className="text-gray-700 font-bold">New Session</h2>
        <AddIcon className="text-black" />
      </div>
      <div className="flex-grow"></div>
      <div className="flex items-center px-4 py-4">
        <SettingsIcon className="text-gray-700 mr-2" />
        <span className="text-gray-600">Project Settings</span>
      </div>
    </div>
  );
};

export default Sidebar;
