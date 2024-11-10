import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuList,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";
import { useNavigate } from "react-router-dom";
import { Link as ScrollLink } from "react-scroll";

// Icons
import { Leaf, BarChart2, Activity, Info } from "lucide-react";

const Navigation = () => {
  return (
    <div className="fixed top-0 w-full bg-white/95 dark:bg-gray-900/95 z-50 border-b">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          <span className="text-xl font-bold text-green-600">EcoScore</span>
          <NavigationMenu>
            <NavigationMenuList className="hidden md:flex space-x-4">
              {[
                "Features",
                "About",
                "Services",
                "Testimonials",
                "Team",
                "Contact",
              ].map((item) => (
                <NavigationMenuItem key={item}>
                  <ScrollLink
                    to={item.toLowerCase()}
                    smooth={true}
                    duration={500}
                    className={navigationMenuTriggerStyle()}
                  >
                    {item}
                  </ScrollLink>
                </NavigationMenuItem>
              ))}
            </NavigationMenuList>
          </NavigationMenu>
        </div>
      </div>
    </div>
  );
};

const Header = () => {
  const navigate = useNavigate();

  return (
    <section
      id="header"
      className="min-h-screen flex flex-col justify-center items-center text-center"
    >
      <div className="container mx-auto px-4">
        <h1 className="text-5xl md:text-6xl font-bold text-gray-900 dark:text-white mb-6">
          Welcome to <span className="text-green-600">EcoScore</span>
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
          Measure and improve your environmental impact with our innovative
          scoring system
        </p>
        <Button
          size="lg"
          className="bg-green-600 hover:bg-green-700"
          onClick={() => navigate("/sign-in")}
        >
          Get Started
        </Button>
      </div>
    </section>
  );
};

const Features = () => {
  const features = [
    {
      icon: <Leaf className="h-6 w-6" />,
      title: "Eco-Friendly Metrics",
      description:
        "Track your environmental impact with our comprehensive scoring system",
    },
    {
      icon: <BarChart2 className="h-6 w-6" />,
      title: "Real-Time Analytics",
      description: "Monitor your progress with detailed analytics and insights",
    },
    {
      icon: <Activity className="h-6 w-6" />,
      title: "Performance Tracking",
      description: "Set goals and track your improvement over time",
    },
  ];

  return (
    <section id="features" className="py-20 bg-gray-50 dark:bg-gray-800">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl font-bold text-center mb-12">Features</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <Card key={index} className="border-none shadow-lg">
              <CardHeader>
                <div className="w-12 h-12 rounded-full bg-green-100 dark:bg-green-900 flex items-center justify-center mb-4">
                  {feature.icon}
                </div>
                <CardTitle>{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600 dark:text-gray-300">
                  {feature.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

const About = () => {
  return (
    <section id="about" className="py-20">
      <div className="container mx-auto px-4">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div>
            <h2 className="text-3xl font-bold mb-6">About EcoScore</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-6">
              EcoScore is dedicated to helping individuals and businesses
              understand and improve their environmental impact through our
              innovative scoring system.
            </p>
            <Button
              variant="outline"
              className="border-green-600 text-green-600"
            >
              Learn More
            </Button>
          </div>
          <Card className="bg-green-50 dark:bg-green-900/20">
            <CardContent className="p-6">
              <div className="aspect-video bg-green-100 dark:bg-green-900 rounded-lg flex items-center justify-center">
                <Info className="h-12 w-12 text-green-600" />
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};

const Services = () => {
  const services = [
    {
      title: "Environmental Assessment",
      description: "Comprehensive evaluation of your environmental impact",
    },
    {
      title: "Customized Solutions",
      description: "Tailored recommendations for improvement",
    },
    {
      title: "Continuous Monitoring",
      description: "Regular tracking and reporting of your progress",
    },
  ];

  return (
    <section id="services" className="py-20 bg-gray-50 dark:bg-gray-800">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl font-bold text-center mb-12">Our Services</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {services.map((service, index) => (
            <Card key={index}>
              <CardHeader>
                <CardTitle>{service.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600 dark:text-gray-300">
                  {service.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

const Testimonials = () => {
  const testimonials = [
    {
      text: "EcoScore has transformed how we think about environmental impact.",
      author: "Jane Smith",
      role: "CEO, Green Tech",
    },
    {
      text: "The insights provided by EcoScore are invaluable for our business.",
      author: "John Doe",
      role: "Sustainability Director",
    },
    {
      text: "The insights provided by EcoScore are invaluable for our business.",
      author: "Loid Forger",
      role: "Sustainability Director"
    }
  ];

  return (
    <section id="testimonials" className="py-20">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl font-bold text-center mb-12">Testimonials</h2>
        <div className="grid md:grid-cols-2 gap-8">
          {testimonials.map((testimonial, index) => (
            <Card key={index} className="bg-green-50 dark:bg-green-900/20">
              <CardContent className="p-6">
                <p className="italic mb-4">{testimonial.text}</p>
                <div className="flex items-center">
                  <div className="ml-4">
                    <p className="font-semibold">{testimonial.author}</p>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      {testimonial.role}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

const Contact = () => {
  return (
    <section id="contact" className="py-20 bg-gray-50 dark:bg-gray-800">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl font-bold text-center mb-12">Contact Us</h2>
        <Card className="max-w-lg mx-auto">
          <CardContent className="p-6">
            <p className="text-center mb-6">
              Get in touch with us for more information about EcoScore
            </p>
            <Button className="w-full bg-green-600 hover:bg-green-700">
              Contact Us
            </Button>
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-white dark:bg-gray-900">
      <Navigation />
      <Header />
      <Features />
      <About />
      <Services />
      <Testimonials />
      <Contact />
    </div>
  );
};

export default LandingPage;

