import { IconRobot } from "@tabler/icons-react";
import { House, ChartPie, Dumbbell, Trophy, Swords } from "lucide-react";

export const sidebarLinks = [
  {
    label: "Dashboard",
    href: "/dashboard",
    icon: (
      <House className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" />
    ),
  },
  {
    label: "Analytics",
    href: "/analytics",
    icon: (
      <ChartPie className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" />
    ),
  },
  {
    label: "Challenges",
    href: "/challenges",
    icon: (
      <Dumbbell className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" />
    ),
  },
  {
    label: "Leaderboard",
    href: "/leaderboard",
    icon: (
      <Trophy className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" />
    ),
  },
  {
    label: "Chat With Eco",
    href: "/chatbot",
    icon: (
      <IconRobot className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" />
    ),
  },
];

export const userName = "Abhi9";
