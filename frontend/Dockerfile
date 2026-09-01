# Frontend image. Build from frontend/:
#   docker build --build-arg VITE_API_KEY=... -t risk-manager-frontend .
#
# VITE_API_KEY MUST be passed as a build arg, not a container runtime env
# var: Vite substitutes import.meta.env.VITE_* into the JS bundle at
# `npm run build` time, so by the time this container is actually
# running, the value is already baked into the static files — setting
# an `environment:` entry on the running container does nothing. See
# docker-compose.yml's `build.args` for how this gets wired through.
FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_KEY
ENV VITE_API_KEY=$VITE_API_KEY
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
