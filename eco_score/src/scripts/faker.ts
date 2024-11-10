import 'dotenv/config';
import { drizzle } from 'drizzle-orm/node-postgres';
import { faker } from '@faker-js/faker';

const db = drizzle(process.env.DATABASE_URL!);
