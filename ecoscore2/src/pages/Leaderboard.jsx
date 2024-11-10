import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Users, Award } from "lucide-react";
import PdSidebar from "@/components/pd-dashboard/PdSidebar";
import leaderboardData from "@/data/leaderboardData.json";

const Leaderboard = () => {
  const getIcon = (iconName) => {
    switch (iconName) {
      case "Users":
        return <Users className="h-6 w-6 text-green-600" />;
      default:
        return null;
    }
  };

  const topThree = leaderboardData.slice(0, 3);
  const remainingUsers = leaderboardData.slice(3);

  return (
    <div className="min-h-screen h-screen overflow-hidden bg-gray-50 dark:bg-gray-900 flex">
      <PdSidebar />
      <div className="flex-1 p-8 overflow-y-auto">
        <div className="container mx-auto">
          <div className="flex flex-col items-center space-y-4">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              Leaderboard
            </h1>
            <div className="grid w-full max-w-md grid-cols-1 gap-4">
              <Button variant="solid" className="bg-green-600 text-white">
                View All
              </Button>
            </div>
          </div>

          {/* Top 3 Users */}
          <div className="space-y-8 mt-8">
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {topThree.map((user) => (
                <Card
                  key={user.id}
                  className="hover:shadow-lg transition-shadow"
                >
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      {getIcon(user.icon)}
                      <Badge
                        variant="outline"
                        className="bg-amber-100 dark:bg-amber-900"
                      >
                        <Award className="h-4 w-4 mr-1" />
                        Rank #{user.rank}
                      </Badge>
                    </div>
                    <CardTitle className="mt-4">{user.name}</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">
                        Points
                      </span>
                      <span className="text-gray-600 dark:text-gray-400">
                        {user.points}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* Remaining Users */}
          <div className="mt-12">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              All Participants
            </h2>
            <div className="space-y-4">
              {remainingUsers.map((user) => (
                <div
                  key={user.id}
                  className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 rounded-lg shadow-md"
                >
                  <div className="flex items-center space-x-4">
                    {getIcon(user.icon)}
                    <div>
                      <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                        {user.name}
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Rank {user.rank}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-medium text-gray-900 dark:text-white">
                      {user.points} Points
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Leaderboard;
