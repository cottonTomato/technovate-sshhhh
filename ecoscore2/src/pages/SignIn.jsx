import { SignIn as ClerkSignIn } from "@clerk/clerk-react";

export default function SignIn() {
  return (
    <div className="flex justify-center items-center h-screen w-full">
      <ClerkSignIn redirectUrl="/dashboard" signUpUrl="/sign-up" />
    </div>
  );
}
