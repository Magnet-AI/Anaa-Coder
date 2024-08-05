import React from "react";
import { Box, CircularProgress } from "@mui/material";

const LoadingScreen = () => {
  return (
    <Box
      className="flex items-center justify-center h-full"
      sx={{
        backgroundColor: "rgba(255, 255, 255, 0.8)",
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 1000,
      }}
    >
      <CircularProgress size={60} color="primary" />
    </Box>
  );
};

export default LoadingScreen;
