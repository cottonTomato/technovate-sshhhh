import { drizzle } from 'drizzle-orm/node-postgres';
import { databaseURI } from '../common/config';

export const db = drizzle(databaseURI);
