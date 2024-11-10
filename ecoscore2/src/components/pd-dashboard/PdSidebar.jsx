import { useState } from "react";
import { Sidebar, SidebarBody, SidebarLink } from "@/components/ui/sidebar";
import LeafIcon from "@/assets/leaf.svg";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { sidebarLinks } from "@/data/data.jsx";
import { UserButton } from "@clerk/clerk-react";

export default function PdSidebar() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Sidebar open={open} setOpen={setOpen} animate={false}>
        <SidebarBody className="justify-between gap-10">
          <div className="flex flex-col flex-1 overflow-y-auto overflow-x-hidden">
            <>
              <Logo />
            </>
            <div className="mt-8 flex flex-col gap-2">
              {sidebarLinks.map((link, idx) => (
                <SidebarLink key={idx} link={link} />
              ))}
            </div>
          </div>
          <div>
            <UserButton
              showName
              appearance={{
                elements: { userButtonBox: { flexDirection: "row-reverse" } },
              }}
            />
          </div>
        </SidebarBody>
      </Sidebar>
    </>
  );
}

export const Logo = () => {
  return (
    <Link
      to="/dashboard"
      className="font-normal flex space-x-2 items-center text-2xl text-black py-1 relative z-20"
    >
      <img
        src={LeafIcon}
        className="h-8 w-8"
        alt="fuck u u dont get one if u read this msg ur gay"
      />
      <motion.span
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="font-large text-black dark:text-white whitespace-pre"
      >
        Eco Score
      </motion.span>
    </Link>
  );
};
