import React, { useEffect, useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Bell } from "lucide-react";

const notifs = {
  notifications: [
    {
      message: "Energy consumption dropped sharply from the average of 306.15 on 2024-10-10 00:00:00. The value recorded was 23.41."
    },
    {
      message: "Energy consumption spiked significantly above the average of 306.15 on 2024-10-12 00:00:00. The value recorded was 329.75."
    },
    {
      message: "Energy consumption dropped slightly below the average of 306.15 on 2024-10-27 00:00:00. The value recorded was 298.79."
    },
    {
      message: "Energy consumption dropped significantly below the average of 306.15 on 2024-11-09 00:00:00. The value recorded was 284.85."
    }
  ]
};

const NotificationComponent = () => {
  const [currentNotification, setCurrentNotification] = useState(null);
  const [notificationIndex, setNotificationIndex] = useState(0);
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    setCurrentNotification(notifs.notifications[0].message);
    setIsVisible(true);

    const hideTimer = setTimeout(() => {
      setIsVisible(false);
    }, 60 * 1000); 

    const updateInterval = setInterval(() => {
      setNotificationIndex((prevIndex) => {
        const newIndex = (prevIndex + 1) % notifs.notifications.length;
        setCurrentNotification(notifs.notifications[newIndex].message);
        setIsVisible(true);
        return newIndex;
      });

      setTimeout(() => {
        setIsVisible(false);
      }, 60 * 1000);
    }, 10 * 60 * 1000); 

    return () => {
      clearTimeout(hideTimer);
      clearInterval(updateInterval);
    };
  }, []);

  if (!isVisible || !currentNotification) {
    return null;
  }

  return (
    <div className="fixed top-4 right-4 z-50 max-w-md animate-in fade-in slide-in-from-top-2">
      <Card className="bg-white shadow-lg">
        <CardContent className="p-4">
          <div className="flex items-start space-x-4">
            <Bell className="h-5 w-5 text-blue-500 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm text-gray-700">{currentNotification}</p>
            </div>
            <button
              onClick={() => setIsVisible(false)}
              className="text-gray-500 hover:text-gray-700 focus:outline-none"
            >
              &times;
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default NotificationComponent;